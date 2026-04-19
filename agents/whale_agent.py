"""
WhaleAgent — mirrors 47 target wallets and generates trade signals
when a tracked wallet enters a market.

Target wallets are loaded from data/wallet_analysis.json (produced by
main.py --analyze-wallets).  The CLOB API is polled for recent trades;
any match triggers a BUY signal queued with a 60-second delay so the
bot always buys *after* the whale, not simultaneously.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

CLOB_URL = "https://clob.polymarket.com"
WHALE_DELAY_SECONDS = 60


class WhaleAgent:
    """
    Watches top-wallet activity and generates BUY/PASS signals.

    Usage:
        agent = WhaleAgent(wallet_analysis_path=settings.wallet_analysis_path)
        result = await agent.evaluate(market)
    """

    def __init__(
        self,
        wallet_analysis_path: Path | str,
        clob_url: str = CLOB_URL,
        max_wallets: int = 47,
    ) -> None:
        self.wallet_analysis_path = Path(wallet_analysis_path)
        self.clob_url = clob_url
        self.max_wallets = max_wallets
        self._target_wallets: list[str] = []
        self._signal_queue: dict[str, dict[str, Any]] = {}  # market_id → signal
        self._session: aiohttp.ClientSession | None = None
        self._load_target_wallets()

    # ── Wallet loading ─────────────────────────────────────────────────────

    def _load_target_wallets(self) -> None:
        """Load top wallet addresses from wallet_analysis.json."""
        if not self.wallet_analysis_path.exists():
            logger.warning(
                "wallet_analysis.json not found at %s — WhaleAgent has no targets",
                self.wallet_analysis_path,
            )
            return
        try:
            data = json.loads(self.wallet_analysis_path.read_text(encoding="utf-8"))
            top_wallets: list[dict[str, Any]] = data.get("top_wallets", [])
            self._target_wallets = [
                str(w.get("wallet", "")) for w in top_wallets if w.get("wallet")
            ][: self.max_wallets]
            logger.info(
                "WhaleAgent loaded %d target wallets", len(self._target_wallets)
            )
        except Exception as exc:
            logger.error("Failed to load wallet analysis: %s", exc)

    # ── HTTP session ───────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── CLOB queries ───────────────────────────────────────────────────────

    async def _fetch_recent_trades(self, market_id: str) -> list[dict[str, Any]]:
        """Fetch recent trades for a market from the CLOB API."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.clob_url}/trades",
                params={"market": market_id, "limit": "100"},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list):
                        return data
                    return data.get("data", data.get("trades", []))
        except Exception as exc:
            logger.debug("Failed to fetch trades for %s: %s", market_id, exc)
        return []

    # ── Core logic ─────────────────────────────────────────────────────────

    async def check_whale_activity(self, market_id: str) -> list[dict[str, Any]]:
        """
        Return a list of recent trades by target wallets in this market.
        Each entry: {"wallet": str, "side": str, "price": float, "ts": float}
        """
        trades = await self._fetch_recent_trades(market_id)
        whale_trades: list[dict[str, Any]] = []
        target_set = set(self._target_wallets)

        for trade in trades:
            maker = str(trade.get("maker", trade.get("makerAddress", "")))
            taker = str(trade.get("taker", trade.get("takerAddress", "")))
            wallet = None
            if maker in target_set:
                wallet = maker
            elif taker in target_set:
                wallet = taker

            if wallet:
                whale_trades.append(
                    {
                        "wallet": wallet,
                        "side": str(trade.get("side", trade.get("type", "BUY"))).upper(),
                        "price": float(trade.get("price", 0.5)),
                        "ts": float(trade.get("timestamp", trade.get("createdAt", time.time()))),
                    }
                )

        return whale_trades

    async def get_active_whale_markets(self) -> list[str]:
        """
        Return market IDs where at least one target wallet has traded
        in the last hour and a signal is queued.
        """
        now = time.time()
        active: list[str] = []
        for market_id, signal in list(self._signal_queue.items()):
            queue_ts = signal.get("queued_at", 0)
            # Signal is "active" if queued within the last 60 minutes
            if now - queue_ts < 3600:
                active.append(market_id)
            else:
                # Expire old signals
                del self._signal_queue[market_id]
        return active

    async def evaluate(self, market: Any) -> dict[str, Any]:
        """
        Evaluate a market for whale activity.

        Returns:
            {"action": "BUY"|"SELL"|"PASS", "confidence": float, "reason": str}
        """
        market_id: str = getattr(market, "condition_id", str(market))

        if not self._target_wallets:
            return {"action": "PASS", "confidence": 0.0, "reason": "no_target_wallets"}

        whale_trades = await self.check_whale_activity(market_id)
        if not whale_trades:
            return {"action": "PASS", "confidence": 0.0, "reason": "no_whale_activity"}

        # Count recent buys vs sells (last 2 hours)
        cutoff = time.time() - 7200
        recent = [t for t in whale_trades if t["ts"] >= cutoff]
        if not recent:
            return {"action": "PASS", "confidence": 0.0, "reason": "whale_activity_stale"}

        buys = sum(1 for t in recent if t["side"] in ("BUY", "YES"))
        sells = sum(1 for t in recent if t["side"] in ("SELL", "NO"))
        unique_wallets = len({t["wallet"] for t in recent})

        # Determine action
        if buys > sells:
            action = "BUY"
            confidence = min(0.95, 0.5 + unique_wallets * 0.1 + buys * 0.05)
            reason = f"{unique_wallets} whale(s) buying ({buys}B/{sells}S), delay={WHALE_DELAY_SECONDS}s"
            # Queue with delay
            self._signal_queue[market_id] = {
                "action": action,
                "queued_at": time.time(),
                "execute_after": time.time() + WHALE_DELAY_SECONDS,
            }
        elif sells > buys:
            action = "SELL"
            confidence = min(0.90, 0.4 + unique_wallets * 0.1)
            reason = f"{unique_wallets} whale(s) selling ({buys}B/{sells}S)"
        else:
            action = "PASS"
            confidence = 0.3
            reason = f"mixed whale signals ({buys}B/{sells}S)"

        logger.info(
            "WhaleAgent %s: %s (conf=%.2f) — %s",
            market_id[:20], action, confidence, reason,
        )
        return {"action": action, "confidence": confidence, "reason": reason}
