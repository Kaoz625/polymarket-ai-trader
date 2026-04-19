"""
ConvergenceAgent — enters when the market price moves toward the AI's
fair-value estimate.

Strategy: if the current YES price has moved >5% closer to the AI's
fair_value_yes in the last 4 hours, the market is "buying the dip" —
the crowd is correcting toward true probability and momentum favours entry.

Price history is stored in SQLite (table: price_history).
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Minimum convergence toward fair value (in absolute price units) to signal
MIN_CONVERGENCE = 0.05
# Look-back window for convergence measurement
LOOKBACK_SECONDS = 4 * 3600  # 4 hours


class ConvergenceAgent:
    """
    Generates BUY signals when a market is converging toward the AI's
    fair-value estimate (buying the dip).

    Usage:
        agent = ConvergenceAgent(db_path=settings.db_path)
        result = agent.evaluate(market, fair_value_yes=0.65)
    """

    def __init__(self, db_path: str | Path = "data/trading.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    # ── Database ───────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT NOT NULL,
                    price REAL NOT NULL,
                    ts REAL NOT NULL
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_ph_market_ts "
                "ON price_history (market_id, ts)"
            )

    def _record_price(self, market_id: str, price: float) -> None:
        """Insert a price observation into price_history."""
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO price_history (market_id, price, ts) VALUES (?, ?, ?)",
                (market_id, price, time.time()),
            )

    def _get_price_lookback(
        self, market_id: str, lookback_seconds: float = LOOKBACK_SECONDS
    ) -> list[tuple[float, float]]:
        """Return (ts, price) rows for market_id in the lookback window."""
        cutoff = time.time() - lookback_seconds
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT ts, price FROM price_history "
                "WHERE market_id = ? AND ts >= ? ORDER BY ts ASC",
                (market_id, cutoff),
            ).fetchall()
        return [(float(r[0]), float(r[1])) for r in rows]

    # ── Convergence scoring ────────────────────────────────────────────────

    def _measure_convergence(
        self,
        market_id: str,
        current_price: float,
        fair_value: float,
    ) -> float:
        """
        Return how much the price has moved toward fair_value in the last 4h.

        Positive → price moved toward fair value (good).
        Negative → price moved away from fair value.
        0        → no history or no movement.
        """
        history = self._get_price_lookback(market_id)
        if not history:
            return 0.0

        oldest_price = history[0][1]

        # Distance from fair value at start of window vs now
        old_distance = abs(oldest_price - fair_value)
        new_distance = abs(current_price - fair_value)

        # Positive = converged, negative = diverged
        return round(old_distance - new_distance, 6)

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        market: Any,
        fair_value_yes: float | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a market for price convergence toward AI fair value.

        Args:
            market:         Market object with .condition_id and .yes_price attrs
                            OR a dict with the same keys.
            fair_value_yes: AI's estimate of true YES probability (0-1).
                            If not provided returns PASS.

        Returns:
            {"action": "BUY"|"SELL"|"PASS", "confidence": float, "reason": str}
        """
        if isinstance(market, dict):
            market_id = str(market.get("condition_id", market.get("market_id", "")))
            current_price = float(market.get("yes_price", 0.5))
        else:
            market_id = str(getattr(market, "condition_id", ""))
            current_price = float(getattr(market, "yes_price", 0.5))

        if not market_id:
            return {"action": "PASS", "confidence": 0.0, "reason": "no_market_id"}

        if fair_value_yes is None:
            return {"action": "PASS", "confidence": 0.0, "reason": "no_fair_value"}

        # Always record the current price for future lookbacks
        self._record_price(market_id, current_price)

        convergence = self._measure_convergence(market_id, current_price, fair_value_yes)

        # Need sufficient convergence to signal
        if convergence < MIN_CONVERGENCE:
            reason = (
                f"convergence={convergence:.3f} below threshold={MIN_CONVERGENCE}"
            )
            return {"action": "PASS", "confidence": 0.0, "reason": reason}

        # Determine direction: if fair_value > current_price, we expect further rise
        edge = fair_value_yes - current_price
        if edge > 0.05:
            action = "BUY"
        elif edge < -0.05:
            action = "SELL"
        else:
            return {
                "action": "PASS",
                "confidence": 0.1,
                "reason": f"converging but edge too thin (gap={edge:.3f})",
            }

        # Confidence scales with convergence magnitude and remaining edge
        confidence = min(0.95, 0.5 + convergence * 3.0 + abs(edge) * 1.0)

        reason = (
            f"price converged {convergence:.3f} toward fair_value={fair_value_yes:.3f} "
            f"in last 4h; current={current_price:.3f}, edge={edge:+.3f}"
        )
        logger.info(
            "ConvergenceAgent %s: %s (conf=%.2f) — %s",
            market_id[:20], action, confidence, reason,
        )
        return {"action": action, "confidence": confidence, "reason": reason}
