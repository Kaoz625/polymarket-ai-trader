"""
ScorerAgent — fetches all active Polymarket markets, scores each one
0-100 based on liquidity, spread, volume trend, time-to-resolution,
and sentiment, then returns the top candidates.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

CLOB_URL = "https://clob.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com"

# Score weight distribution (must sum to 100)
WEIGHT_LIQUIDITY = 30
WEIGHT_SPREAD = 20
WEIGHT_VOLUME_TREND = 20
WEIGHT_TIME_TO_RES = 15
WEIGHT_SENTIMENT = 15


@dataclass
class Market:
    condition_id: str
    question: str
    yes_price: float
    no_price: float
    volume_24h: float
    open_interest: float
    liquidity: float
    spread: float
    resolution_ts: float  # unix seconds; 0 if unknown
    tokens: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def time_to_resolution_days(self) -> float:
        if self.resolution_ts <= 0:
            return 999.0
        return max(0.0, (self.resolution_ts - time.time()) / 86400)


@dataclass
class ScoredMarket:
    market: Market
    score: float
    score_breakdown: dict[str, float]


class ScorerAgent:
    """Fetches and scores Polymarket markets."""

    def __init__(
        self,
        min_liquidity: float = 1000.0,
        clob_url: str = CLOB_URL,
        gamma_url: str = GAMMA_URL,
    ) -> None:
        self.min_liquidity = min_liquidity
        self.clob_url = clob_url
        self.gamma_url = gamma_url
        self._session: aiohttp.ClientSession | None = None

    # ── HTTP session ───────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        session = await self._get_session()
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ── Market fetching ────────────────────────────────────────────────────

    async def fetch_markets(self, limit: int = 500) -> list[Market]:
        """
        Fetch active markets from the Gamma API (richer metadata) and
        enrich with CLOB pricing data.
        Returns only markets with liquidity >= min_liquidity.
        """
        markets: list[Market] = []
        offset = 0
        page_size = min(limit, 100)

        while len(markets) < limit:
            try:
                data = await self._get(
                    f"{self.gamma_url}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": page_size,
                        "offset": offset,
                    },
                )
            except Exception as exc:
                logger.warning("Gamma API error at offset %d: %s", offset, exc)
                break

            if not data:
                break

            raw_list: list[dict[str, Any]] = (
                data if isinstance(data, list) else data.get("markets", data.get("data", []))
            )
            if not raw_list:
                break

            for raw in raw_list:
                market = self._parse_gamma_market(raw)
                if market and market.liquidity >= self.min_liquidity:
                    markets.append(market)

            if len(raw_list) < page_size:
                break
            offset += page_size

        logger.info("Fetched %d qualifying markets", len(markets))
        return markets[:limit]

    @staticmethod
    def _parse_gamma_market(raw: dict[str, Any]) -> Market | None:
        """Convert a Gamma API market dict into a Market dataclass."""
        try:
            condition_id = raw.get("conditionId") or raw.get("id") or raw.get("marketMakerAddress", "")
            question = raw.get("question", raw.get("title", ""))
            if not condition_id or not question:
                return None

            tokens: list[dict[str, Any]] = raw.get("tokens", raw.get("outcomes", []))
            yes_price = 0.5
            no_price = 0.5
            if tokens:
                for tok in tokens:
                    name = str(tok.get("outcome", tok.get("name", ""))).upper()
                    price = float(tok.get("price", tok.get("lastTradePrice", 0.5)))
                    if name in ("YES", "TRUE", "1"):
                        yes_price = price
                    elif name in ("NO", "FALSE", "0"):
                        no_price = price

            volume_24h = float(raw.get("volume24hr", raw.get("volume", 0)))
            liquidity = float(raw.get("liquidity", raw.get("liquidityNum", 0)))
            open_interest = float(raw.get("openInterest", raw.get("usdcSize", 0)))

            # Spread = ask - bid; if we only have last price use a proxy
            spread = abs(1.0 - yes_price - no_price)

            res_ts = 0.0
            for key in ("endDate", "resolutionTime", "resolution_time", "closeTime"):
                val = raw.get(key)
                if val:
                    try:
                        from datetime import datetime, timezone
                        if isinstance(val, (int, float)):
                            res_ts = float(val)
                        else:
                            dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                            res_ts = dt.replace(tzinfo=timezone.utc).timestamp()
                        break
                    except Exception:
                        pass

            return Market(
                condition_id=condition_id,
                question=question,
                yes_price=yes_price,
                no_price=no_price,
                volume_24h=volume_24h,
                open_interest=open_interest,
                liquidity=liquidity,
                spread=spread,
                resolution_ts=res_ts,
                tokens=tokens,
                raw=raw,
            )
        except Exception as exc:
            logger.debug("Failed to parse market %s: %s", raw.get("id"), exc)
            return None

    # ── Orderbook analysis ─────────────────────────────────────────────────

    async def analyze_orderbook(self, market_id: str) -> dict[str, Any]:
        """
        Fetch the CLOB orderbook for a market and return depth metrics.
        """
        try:
            data = await self._get(
                f"{self.clob_url}/book",
                params={"token_id": market_id},
            )
            bids: list[dict[str, Any]] = data.get("bids", [])
            asks: list[dict[str, Any]] = data.get("asks", [])

            def _total_size(side: list[dict[str, Any]]) -> float:
                return sum(float(o.get("size", 0)) for o in side)

            bid_depth = _total_size(bids)
            ask_depth = _total_size(asks)
            best_bid = float(bids[0]["price"]) if bids else 0.0
            best_ask = float(asks[0]["price"]) if asks else 1.0
            spread = best_ask - best_bid

            return {
                "market_id": market_id,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "total_depth": bid_depth + ask_depth,
                "imbalance": (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1),
            }
        except Exception as exc:
            logger.warning("Orderbook fetch failed for %s: %s", market_id, exc)
            return {}

    # ── Scoring ────────────────────────────────────────────────────────────

    def score_market(self, market: Market) -> ScoredMarket:
        """
        Score a market 0-100 across five dimensions.

        Liquidity (30 pts):  log-scaled; full marks at $100k+
        Spread    (20 pts):  tighter is better; full marks at <0.5%
        Vol trend (20 pts):  higher 24h volume relative to OI is better
        Time (15 pts):       sweet spot 3-30 days; penalise <1d and >90d
        Sentiment (15 pts):  price not at extremes (0.15–0.85 range)
        """
        breakdown: dict[str, float] = {}

        # 1. Liquidity
        import math
        liq = max(market.liquidity, 1.0)
        liq_score = min(WEIGHT_LIQUIDITY, WEIGHT_LIQUIDITY * math.log10(liq) / 5)
        breakdown["liquidity"] = round(liq_score, 2)

        # 2. Spread
        spread_pct = market.spread
        if spread_pct <= 0.005:
            spread_score = float(WEIGHT_SPREAD)
        elif spread_pct >= 0.20:
            spread_score = 0.0
        else:
            spread_score = WEIGHT_SPREAD * (1 - (spread_pct - 0.005) / 0.195)
        breakdown["spread"] = round(spread_score, 2)

        # 3. Volume trend
        oi = max(market.open_interest, 1.0)
        vol_ratio = market.volume_24h / oi
        vol_score = min(WEIGHT_VOLUME_TREND, WEIGHT_VOLUME_TREND * vol_ratio * 5)
        breakdown["volume_trend"] = round(vol_score, 2)

        # 4. Time to resolution
        days = market.time_to_resolution_days
        if days <= 0:
            time_score = 0.0
        elif days < 1:
            time_score = WEIGHT_TIME_TO_RES * (days / 1)
        elif 3 <= days <= 30:
            time_score = float(WEIGHT_TIME_TO_RES)
        elif days <= 90:
            time_score = WEIGHT_TIME_TO_RES * (1 - (days - 30) / 60) * 0.5 + WEIGHT_TIME_TO_RES * 0.5
        else:
            time_score = WEIGHT_TIME_TO_RES * 0.2
        breakdown["time_to_resolution"] = round(time_score, 2)

        # 5. Sentiment (price in interesting range)
        yes = market.yes_price
        if 0.15 <= yes <= 0.85:
            # Most interesting when neither near 0 nor near 1
            distance_from_extreme = min(yes - 0.15, 0.85 - yes) / 0.35
            sentiment_score = WEIGHT_SENTIMENT * min(1.0, distance_from_extreme * 2)
        else:
            sentiment_score = 0.0
        breakdown["sentiment"] = round(sentiment_score, 2)

        total = sum(breakdown.values())
        return ScoredMarket(market=market, score=round(total, 2), score_breakdown=breakdown)

    async def get_top_markets(self, n: int = 20, min_score: float = 70.0) -> list[ScoredMarket]:
        """
        Fetch all active markets, score them, return the top-n above min_score.
        """
        markets = await self.fetch_markets(limit=500)
        scored = [self.score_market(m) for m in markets]
        scored.sort(key=lambda s: s.score, reverse=True)
        top = [s for s in scored if s.score >= min_score]
        logger.info(
            "Scored %d markets; %d above %.0f (returning top %d)",
            len(scored), len(top), min_score, n,
        )
        return top[:n]
