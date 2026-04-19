"""
AIAnalyst — uses Anthropic (claude-sonnet-4-6) with automatic OpenAI fallback.
Multiple OpenAI keys are rotated on rate-limit or auth errors.
Responses are cached for 5 minutes to avoid redundant API calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

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


class AIAnalyst:
    """
    Market analyst that tries Anthropic first, then rotates through OpenAI keys.
    """

    ANTHROPIC_MODEL = "claude-sonnet-4-6"
    OPENAI_MODEL = "gpt-4o-mini"

    def __init__(self, anthropic_api_key: str = "", openai_api_keys: list[str] | None = None) -> None:
        self._anthropic_key = anthropic_api_key
        self._openai_keys: list[str] = list(openai_api_keys or [])
        self._openai_key_index = 0
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

        # Lazy-init clients
        self._anthropic_client = None
        self._openai_client = None

        if anthropic_api_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                logger.info("Anthropic client initialized (claude-sonnet-4-6)")
            except Exception as e:
                logger.warning("Anthropic init failed: %s", e)

        if self._openai_keys:
            logger.info("OpenAI fallback ready with %d key(s)", len(self._openai_keys))

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

    # ── Provider calls ─────────────────────────────────────────────────────

    def _call_anthropic(self, user_message: str) -> dict[str, Any] | None:
        if not self._anthropic_client:
            return None
        try:
            import anthropic
            response = self._anthropic_client.messages.create(
                model=self.ANTHROPIC_MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
            return self._parse_json(raw)
        except Exception as exc:
            logger.warning("Anthropic call failed: %s", exc)
            return None

    def _call_openai(self, user_message: str) -> dict[str, Any] | None:
        if not self._openai_keys:
            return None
        # Try each key in rotation
        for attempt in range(len(self._openai_keys)):
            key = self._openai_keys[self._openai_key_index % len(self._openai_keys)]
            try:
                import openai
                client = openai.OpenAI(api_key=key)
                response = client.chat.completions.create(
                    model=self.OPENAI_MODEL,
                    max_tokens=512,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                )
                raw = response.choices[0].message.content.strip()
                logger.debug("OpenAI key #%d succeeded", self._openai_key_index % len(self._openai_keys))
                return self._parse_json(raw)
            except Exception as exc:
                logger.warning(
                    "OpenAI key #%d failed (%s), rotating...",
                    self._openai_key_index % len(self._openai_keys), exc
                )
                self._openai_key_index += 1
        logger.error("All OpenAI keys exhausted")
        return None

    def _call_ai(self, user_message: str) -> dict[str, Any]:
        """Try Anthropic first, fall back to OpenAI key rotation."""
        result = self._call_anthropic(user_message)
        if result is not None:
            return result
        result = self._call_openai(user_message)
        if result is not None:
            return result
        return self._default_analysis()

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    @staticmethod
    def _default_analysis() -> dict[str, Any]:
        return {
            "sentiment": "neutral",
            "confidence": 0.5,
            "fair_value_yes": 0.5,
            "edge_description": "No AI provider available.",
            "key_risks": ["All AI providers unavailable"],
            "recommendation": "pass",
        }

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze_market(self, market_data: dict[str, Any]) -> dict[str, Any]:
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

        result = self._call_ai(user_message)
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
            result = self._call_ai(user_message)
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


# Backwards-compat alias
ClaudeAnalyst = AIAnalyst
