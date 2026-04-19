"""
brain.py — Standalone brain process.

Reads data/queue.json, runs AIAnalyst on each market, applies 4-check logic:
  1. base_rate from wallet_analysis.json
  2. recent news (skip if can't fetch)
  3. whale_check: is any target wallet active in this market?
  4. confidence > 0.75 → write to data/thesis.json

Usage:
    python scripts/brain.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from agents.claude_analyst import AIAnalyst
from agents.whale_agent import WhaleAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("brain")

CONFIDENCE_THRESHOLD = 0.75


def _load_base_rates() -> dict[str, float]:
    """
    Load keyword → base_rate mapping from wallet_analysis.json.
    Returns empty dict if file doesn't exist.
    """
    path = settings.wallet_analysis_path
    if not path.exists():
        logger.warning("wallet_analysis.json not found — skipping base_rate check")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # The patterns section has median win rate which is our base rate proxy
        patterns = data.get("patterns", {})
        median_win_rate = float(patterns.get("median_win_rate", 0.6))
        return {"__default__": median_win_rate}
    except Exception as exc:
        logger.warning("Failed to load base rates: %s", exc)
        return {}


def _get_base_rate(question: str, base_rates: dict[str, float]) -> float:
    """Return base rate for a market question (keyword lookup, falls back to default)."""
    q_lower = question.lower()
    for keyword, rate in base_rates.items():
        if keyword != "__default__" and keyword in q_lower:
            return rate
    return base_rates.get("__default__", 0.6)


async def _fetch_recent_news(question: str) -> str | None:
    """
    Attempt a very simple news fetch using DuckDuckGo instant answers API.
    Returns a summary string or None if fetch fails.
    """
    try:
        import aiohttp
        query = question[:80].replace(" ", "+")
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=8)
        ) as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    abstract = data.get("AbstractText", "")
                    if abstract:
                        return abstract[:500]
        return None
    except Exception as exc:
        logger.debug("News fetch failed: %s", exc)
        return None


async def process_market(
    market: dict,
    analyst: AIAnalyst,
    whale_agent: WhaleAgent,
    base_rates: dict[str, float],
) -> dict | None:
    """
    Run 4-check logic on a single market dict.
    Returns a thesis dict or None if the market doesn't pass.
    """
    question = market.get("question", "")
    market_id = market.get("condition_id", "")

    # Check 1: base rate
    base_rate = _get_base_rate(question, base_rates)
    logger.debug("Market %s base_rate=%.2f", market_id[:20], base_rate)

    # Check 2: recent news (skip market if fetch outright errors — not just empty)
    news_context = await _fetch_recent_news(question)
    # news_context may be None (unavailable) — we proceed but note it

    # Check 3: whale activity
    class _FakeMarket:
        condition_id = market_id

    whale_result = await whale_agent.evaluate(_FakeMarket())
    whale_active = whale_result.get("action") in ("BUY", "SELL")
    whale_confidence = float(whale_result.get("confidence", 0.0))

    # Check 4: AI analysis
    market_data = {
        "question": question,
        "yes_price": market.get("yes_price", 0.5),
        "no_price": market.get("no_price", 0.5),
        "volume_24h": market.get("volume_24h", 0),
        "liquidity": market.get("liquidity", 0),
        "score": market.get("score", 0),
        "days_to_resolve": market.get("time_to_resolution_days", 0),
        "recent_news": news_context or "No recent news found.",
    }
    analysis = analyst.analyze_market(market_data)

    ai_confidence = float(analysis.get("confidence", 0.0))
    recommendation = analysis.get("recommendation", "pass")
    fair_value = float(analysis.get("fair_value_yes", market.get("yes_price", 0.5)))

    # Boost confidence if whale agrees
    combined_confidence = ai_confidence
    if whale_active and whale_result.get("action") == (
        "BUY" if recommendation == "enter_yes" else "SELL"
    ):
        combined_confidence = min(0.98, ai_confidence + whale_confidence * 0.15)
        logger.info("Whale confirmation for %s: conf boosted to %.2f", market_id[:20], combined_confidence)

    # Hard threshold
    if combined_confidence < CONFIDENCE_THRESHOLD:
        logger.info(
            "SKIP %s: conf=%.2f < %.2f", question[:50], combined_confidence, CONFIDENCE_THRESHOLD
        )
        return None

    if recommendation == "pass":
        logger.info("SKIP %s: AI recommendation=pass", question[:50])
        return None

    thesis = {
        "condition_id": market_id,
        "question": question,
        "yes_price": market.get("yes_price", 0.5),
        "no_price": market.get("no_price", 0.5),
        "volume_24h": market.get("volume_24h", 0),
        "liquidity": market.get("liquidity", 0),
        "score": market.get("score", 0),
        "resolution_ts": market.get("resolution_ts", 0),
        "time_to_resolution_days": market.get("time_to_resolution_days", 0),
        # AI analysis
        "sentiment": analysis.get("sentiment", "neutral"),
        "ai_confidence": ai_confidence,
        "combined_confidence": combined_confidence,
        "fair_value_yes": fair_value,
        "edge_description": analysis.get("edge_description", ""),
        "key_risks": analysis.get("key_risks", []),
        "recommendation": recommendation,
        # Supporting checks
        "base_rate": base_rate,
        "whale_active": whale_active,
        "whale_action": whale_result.get("action", "PASS"),
        "news_available": news_context is not None,
        "generated_at": time.time(),
    }
    return thesis


async def main() -> None:
    logger.info("Brain starting …")

    queue_path = settings.data_dir / "queue.json"
    if not queue_path.exists():
        logger.error("queue.json not found — run scanner.py first")
        sys.exit(1)

    markets: list[dict] = json.loads(queue_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d markets from queue.json", len(markets))

    analyst = AIAnalyst(
        anthropic_api_key=settings.anthropic_api_key,
        openai_api_keys=settings.openai_api_keys,
    )
    whale_agent = WhaleAgent(wallet_analysis_path=settings.wallet_analysis_path)
    base_rates = _load_base_rates()

    theses: list[dict] = []
    for market in markets:
        try:
            thesis = await process_market(market, analyst, whale_agent, base_rates)
            if thesis:
                theses.append(thesis)
                logger.info(
                    "THESIS: %s  conf=%.2f  rec=%s",
                    market.get("question", "")[:50],
                    thesis["combined_confidence"],
                    thesis["recommendation"],
                )
        except Exception as exc:
            logger.error("Error processing %s: %s", market.get("condition_id", "?"), exc)

    await whale_agent.close()

    thesis_path = settings.data_dir / "thesis.json"
    thesis_path.write_text(json.dumps(theses, indent=2), encoding="utf-8")
    logger.info(
        "Brain complete: %d/%d markets passed → wrote %s",
        len(theses), len(markets), thesis_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
