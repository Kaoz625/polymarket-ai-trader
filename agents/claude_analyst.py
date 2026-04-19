"""
ClaudeAnalyst — uses the Anthropic API (claude-sonnet-4-6) to provide
qualitative market analysis.  Responses are cached for 5 minutes to
avoid redundant API calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 300  # 5 minutes

SYSTEM_PROMPT = """You are an expert prediction market analyst specialising in Polymarket.
Your job is to assess binary-outcome markets and identify trading edge.

When analysing a market you must output a JSON object with exactly these keys:
  sentiment       : "bullish_yes" | "bearish_yes" | "neutral"
  confidence      : float 0-1 (how confident you are in the direction)
  fair_value_yes  : float 0-1 (your estimate of the true YES probability)
  edge_description: string (one concise sentence on why there is edge)
  key_risks       : list of strings (up to 3 risks)
  recommendation  : "enter_yes" | "enter_no" | "pass"

Output ONLY valid JSON. No preamble, no markdown fences."""


class ClaudeAnalyst:
    """
    Thin wrapper around the Anthropic Messages API for market analysis.
    Caches responses keyed by a hash of the market data.
    """

    MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    # ── Cache helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(data: dict[str, Any]) -> str:
        blob = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _get_cached(self, key: str) -> dict[str, Any] | None:
        if key in self._cache:
            ts, result = self._cache[key]
            if time.time() - ts < CACHE_TTL_SECONDS:
                return result
            del self._cache[key]
        return None

    def _set_cache(self, key: str, result: dict[str, Any]) -> None:
        self._cache[key] = (time.time(), result)

    # ── API call ───────────────────────────────────────────────────────────

    def _call_claude(self, user_message: str) -> dict[str, Any]:
        """Send a message to claude-sonnet-4-6 and parse JSON response."""
        try:
            response = self._client.messages.create(
                model=self.MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if the model adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Claude returned non-JSON: %s", exc)
            return self._default_analysis()
        except anthropic.APIError as exc:
            logger.error("Anthropic API error: %s", exc)
            return self._default_analysis()

    @staticmethod
    def _default_analysis() -> dict[str, Any]:
        return {
            "sentiment": "neutral",
            "confidence": 0.5,
            "fair_value_yes": 0.5,
            "edge_description": "Insufficient information to assess edge.",
            "key_risks": ["API unavailable"],
            "recommendation": "pass",
        }

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze_market(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """
        Analyse a single market.

        market_data should contain:
            question       : str
            yes_price      : float
            no_price       : float
            volume_24h     : float
            liquidity      : float
            score          : float (from ScorerAgent)
            days_to_resolve: float (optional)
            recent_news    : str (optional — any context the caller has)
        """
        key = self._cache_key(market_data)
        cached = self._get_cached(key)
        if cached:
            logger.debug("Cache hit for market analysis")
            return cached

        question = market_data.get("question", "Unknown market")
        yes_px = market_data.get("yes_price", 0.5)
        no_px = market_data.get("no_price", 0.5)
        volume = market_data.get("volume_24h", 0)
        liquidity = market_data.get("liquidity", 0)
        score = market_data.get("score", 0)
        days = market_data.get("days_to_resolve", "unknown")
        news = market_data.get("recent_news", "No additional context provided.")

        user_message = (
            f"Analyse this prediction market:\n\n"
            f"Question: {question}\n"
            f"Current YES price: {yes_px:.4f} ({yes_px*100:.1f}%)\n"
            f"Current NO price:  {no_px:.4f} ({no_px*100:.1f}%)\n"
            f"24h volume: ${volume:,.0f}\n"
            f"Liquidity: ${liquidity:,.0f}\n"
            f"Days to resolution: {days}\n"
            f"Algorithmic score: {score:.1f}/100\n\n"
            f"Additional context:\n{news}\n\n"
            f"Output your analysis as JSON."
        )

        result = self._call_claude(user_message)
        self._set_cache(key, result)
        logger.info(
            "Market analysis: %s → %s (conf=%.2f, rec=%s)",
            question[:60],
            result.get("sentiment"),
            result.get("confidence", 0),
            result.get("recommendation"),
        )
        return result

    def summarize_performance(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Ask Claude to summarise recent trading performance and suggest
        improvements.  Returns a dict with keys: summary, suggestions.
        """
        if not trades:
            return {"summary": "No trades to summarise.", "suggestions": []}

        cache_key = self._cache_key({"trades_hash": len(trades), "latest": trades[-1]})
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        wins = sum(1 for t in trades if float(t.get("pnl_usdc", 0)) > 0)
        losses = len(trades) - wins
        total_pnl = sum(float(t.get("pnl_usdc", 0)) for t in trades)
        avg_pnl = total_pnl / len(trades) if trades else 0

        trade_lines = "\n".join(
            f"- {t.get('question', 'unknown')[:60]}: "
            f"pnl={t.get('pnl_usdc', 0):.2f} USDC, "
            f"reason={t.get('exit_reason', 'unknown')}"
            for t in trades[-20:]
        )

        user_message = (
            f"Summarise this recent trading session on Polymarket:\n\n"
            f"Trades: {len(trades)} | Wins: {wins} | Losses: {losses}\n"
            f"Total PnL: ${total_pnl:.2f} | Avg per trade: ${avg_pnl:.2f}\n\n"
            f"Recent trades:\n{trade_lines}\n\n"
            f"Output JSON with keys 'summary' (string) and 'suggestions' (list of strings)."
        )

        try:
            result = self._call_claude(user_message)
            # Normalise keys in case Claude drifts
            if "summary" not in result:
                result["summary"] = str(result)
            if "suggestions" not in result:
                result["suggestions"] = []
        except Exception as exc:
            logger.error("summarize_performance failed: %s", exc)
            result = {
                "summary": f"Win rate: {wins/len(trades)*100:.1f}%. Total PnL: ${total_pnl:.2f}",
                "suggestions": [],
            }

        self._set_cache(cache_key, result)
        return result
