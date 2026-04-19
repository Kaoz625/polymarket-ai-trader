"""
exit_monitor.py — Standalone exit monitor.

Reads open positions from SQLite, checks 3 exit triggers:
  1. 85% target captured
  2. 3× volume spike
  3. 24h stale thesis (price moved < 2% in 24h)

Runs in a loop with 60-second interval.

Usage:
    python scripts/exit_monitor.py
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from agents.strategy_agent import StrategyAgent, Position

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("exit_monitor")

POLL_INTERVAL = 60  # seconds


def _load_open_positions(db_path: str) -> list[Position]:
    """Load all open positions from the executor's open_positions table."""
    positions: list[Position] = []
    try:
        with sqlite3.connect(db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM open_positions ORDER BY opened_at ASC"
            ).fetchall()
        import uuid as _uuid
        for row in rows:
            r = dict(row)
            positions.append(
                Position(
                    position_id=r.get("position_id", str(_uuid.uuid4())),
                    market_id=r.get("market_id", ""),
                    question=r.get("question", ""),
                    side=r.get("side", "YES"),
                    entry_price=float(r.get("entry_price", 0.5)),
                    current_price=float(r.get("entry_price", 0.5)),  # refreshed below
                    size_usdc=float(r.get("size_usdc", 0)),
                    target_price=float(r.get("entry_price", 0.5)) * 1.15,  # fallback
                    stop_price=float(r.get("entry_price", 0.5)) * 0.88,    # fallback
                    entry_ts=float(r.get("opened_at", time.time())),
                    volume_baseline=0.0,
                )
            )
    except Exception as exc:
        logger.error("Failed to load open positions: %s", exc)
    return positions


def _close_position_in_executor_db(
    db_path: str, position_id: str, exit_price: float, exit_reason: str
) -> None:
    """
    Move a position from open_positions to closed_positions in the executor DB.
    """
    now = time.time()
    try:
        with sqlite3.connect(db_path) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                "SELECT * FROM open_positions WHERE position_id=?", (position_id,)
            ).fetchone()
            if not row:
                return
            r = dict(row)
            entry_price = float(r.get("entry_price", exit_price))
            size_usdc = float(r.get("size_usdc", 0))
            side = r.get("side", "YES")
            if side == "YES":
                pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0
            else:
                pnl_pct = (entry_price - exit_price) / entry_price if entry_price else 0
            pnl_usdc = pnl_pct * size_usdc

            con.execute("DELETE FROM open_positions WHERE position_id=?", (position_id,))
            con.execute(
                """
                INSERT OR REPLACE INTO closed_positions
                (position_id, market_id, question, side, entry_price, exit_price,
                 size_usdc, pnl_usdc, pnl_pct, exit_reason, opened_at, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position_id, r.get("market_id", ""), r.get("question", ""),
                    side, entry_price, exit_price, size_usdc,
                    round(pnl_usdc, 4), round(pnl_pct, 6),
                    exit_reason, r.get("opened_at", now), now,
                ),
            )
        logger.info(
            "Closed position %s reason=%s pnl=%.2f%%",
            position_id[:16], exit_reason, pnl_pct * 100,
        )
    except Exception as exc:
        logger.error("DB close failed for %s: %s", position_id[:16], exc)


async def check_and_exit_positions(strategy: StrategyAgent, db_path: str) -> None:
    """Load open positions, refresh prices, check exits, close if triggered."""
    positions = _load_open_positions(db_path)
    if not positions:
        logger.debug("No open positions to monitor")
        return

    logger.info("Monitoring %d open position(s)", len(positions))

    for pos in positions:
        try:
            # Refresh live price before exit check
            await strategy.refresh_position_price(pos)

            should_exit, reason = await strategy.should_exit(pos)
            if should_exit:
                logger.info(
                    "EXIT triggered for %s: %s  price=%.4f",
                    pos.question[:50], reason, pos.current_price,
                )
                _close_position_in_executor_db(
                    db_path, pos.position_id, pos.current_price, reason
                )
        except Exception as exc:
            logger.error("Exit check failed for %s: %s", pos.market_id[:20], exc)


async def main() -> None:
    logger.info("Exit monitor starting (interval=%ds) …", POLL_INTERVAL)

    strategy = StrategyAgent(
        exit_threshold=settings.exit_threshold,
        loss_cut=settings.loss_cut,
        volume_spike_multiplier=settings.volume_spike_multiplier,
        max_position_usdc=settings.max_position_size_usdc,
        db_path=str(settings.db_path),
        clob_url=settings.clob_url,
    )

    db_path = str(settings.db_path)

    while True:
        try:
            await check_and_exit_positions(strategy, db_path)
        except Exception as exc:
            logger.error("Monitor cycle error: %s", exc)

        logger.debug("Sleeping %ds …", POLL_INTERVAL)
        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
