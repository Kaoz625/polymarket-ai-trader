"""
scanner.py — Standalone market scanner.

Fetches markets from the Polymarket Gamma API, applies hard-kill filters,
scores them, and writes the top candidates to data/queue.json.

Usage:
    python scripts/scanner.py
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
from agents.scorer_agent import ScorerAgent, ScoredMarket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scanner")

# Hard-kill filter constants
MIN_VOLUME_24H = 50_000       # USDC — slippage kills edge below this
MIN_HOURS_TO_RESOLVE = 4      # too close to resolution
MAX_HOURS_TO_RESOLVE = 168    # 1 week — too slow
MIN_EDGE_GAP = 0.07           # AI estimate vs market price must exceed this


def apply_hard_filters(scored: ScoredMarket, ai_estimate: float | None = None) -> bool:
    """
    Return True if the market passes all hard-kill filters, False otherwise.

    Filters:
        1. volume_24h >= MIN_VOLUME_24H
        2. MIN_HOURS_TO_RESOLVE <= hours_to_resolve <= MAX_HOURS_TO_RESOLVE
        3. If ai_estimate provided: abs(ai_estimate - market_price) >= MIN_EDGE_GAP
    """
    m = scored.market
    hours_to_resolve = m.time_to_resolution_days * 24

    if m.volume_24h < MIN_VOLUME_24H:
        logger.debug("KILL volume: %s  vol=%.0f", m.question[:50], m.volume_24h)
        return False

    if hours_to_resolve < MIN_HOURS_TO_RESOLVE:
        logger.debug("KILL too_close: %s  h=%.1f", m.question[:50], hours_to_resolve)
        return False

    if m.resolution_ts > 0 and hours_to_resolve > MAX_HOURS_TO_RESOLVE:
        logger.debug("KILL too_far: %s  h=%.1f", m.question[:50], hours_to_resolve)
        return False

    if ai_estimate is not None:
        gap = abs(ai_estimate - m.yes_price)
        if gap < MIN_EDGE_GAP:
            logger.debug(
                "KILL thin_edge: %s  gap=%.3f", m.question[:50], gap
            )
            return False

    return True


async def main() -> None:
    logger.info("Scanner starting …")
    scorer = ScorerAgent(
        min_liquidity=settings.min_liquidity,
        clob_url=settings.clob_url,
        gamma_url=settings.gamma_url,
    )

    try:
        scored_markets = await scorer.get_top_markets(n=100, min_score=60.0)
    finally:
        await scorer.close()

    # Apply hard-kill filters (no AI estimate at this stage)
    passed = [sm for sm in scored_markets if apply_hard_filters(sm)]
    logger.info(
        "Hard filters: %d/%d markets passed", len(passed), len(scored_markets)
    )

    # Serialize to queue.json
    queue = []
    for sm in passed:
        m = sm.market
        queue.append(
            {
                "condition_id": m.condition_id,
                "question": m.question,
                "yes_price": m.yes_price,
                "no_price": m.no_price,
                "volume_24h": m.volume_24h,
                "liquidity": m.liquidity,
                "open_interest": m.open_interest,
                "spread": m.spread,
                "resolution_ts": m.resolution_ts,
                "time_to_resolution_days": m.time_to_resolution_days,
                "score": sm.score,
                "score_breakdown": sm.score_breakdown,
                "scanned_at": time.time(),
            }
        )

    queue_path = settings.data_dir / "queue.json"
    queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    logger.info("Wrote %d markets to %s", len(queue), queue_path)


if __name__ == "__main__":
    asyncio.run(main())
