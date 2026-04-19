"""
StrategyAgent — converts market scores + wallet patterns into trade
signals, tracks open positions, and applies the exit rules:
  • Cut losers at 12% (LOSS_CUT)
  • Exit when 85% of expected move captured (EXIT_THRESHOLD)
  • Exit on 3× volume spike (VOLUME_SPIKE_MULTIPLIER)
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

CLOB_URL = "https://clob.polymarket.com"


@dataclass
class TradeSignal:
    market_id: str
    question: str
    side: str          # "YES" or "NO"
    entry_price: float
    target_price: float   # expected move end point
    stop_price: float     # loss-cut level
    confidence: float     # 0-1
    rationale: str
    score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Position:
    position_id: str
    market_id: str
    question: str
    side: str
    entry_price: float
    current_price: float
    size_usdc: float
    target_price: float
    stop_price: float
    entry_ts: float = field(default_factory=time.time)
    volume_baseline: float = 0.0  # rolling avg volume at entry

    @property
    def pnl_pct(self) -> float:
        if self.side == "YES":
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price

    @property
    def pnl_usdc(self) -> float:
        return self.pnl_pct * self.size_usdc

    @property
    def hold_hours(self) -> float:
        return (time.time() - self.entry_ts) / 3600


class StrategyAgent:
    """Generates trade signals and manages exit logic."""

    def __init__(
        self,
        exit_threshold: float = 0.85,
        loss_cut: float = 0.12,
        volume_spike_multiplier: float = 3.0,
        max_position_usdc: float = 100.0,
        db_path: str = "data/trading.db",
        clob_url: str = CLOB_URL,
    ) -> None:
        self.exit_threshold = exit_threshold
        self.loss_cut = loss_cut
        self.volume_spike_multiplier = volume_spike_multiplier
        self.max_position_usdc = max_position_usdc
        self.db_path = db_path
        self.clob_url = clob_url
        self._open_positions: dict[str, Position] = {}
        self._volume_history: dict[str, list[tuple[float, float]]] = {}
        self._session: aiohttp.ClientSession | None = None
        self._init_db()

    # ── DB setup ───────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    market_id TEXT,
                    question TEXT,
                    side TEXT,
                    entry_price REAL,
                    current_price REAL,
                    size_usdc REAL,
                    target_price REAL,
                    stop_price REAL,
                    entry_ts REAL,
                    volume_baseline REAL,
                    closed INTEGER DEFAULT 0,
                    exit_price REAL,
                    exit_reason TEXT,
                    exit_ts REAL
                );
                """
            )

    def _save_position(self, pos: Position) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                INSERT OR REPLACE INTO positions
                (position_id, market_id, question, side, entry_price, current_price,
                 size_usdc, target_price, stop_price, entry_ts, volume_baseline)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pos.position_id, pos.market_id, pos.question, pos.side,
                    pos.entry_price, pos.current_price, pos.size_usdc,
                    pos.target_price, pos.stop_price, pos.entry_ts, pos.volume_baseline,
                ),
            )

    def _close_position_db(
        self, position_id: str, exit_price: float, exit_reason: str
    ) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                UPDATE positions
                SET closed=1, exit_price=?, exit_reason=?, exit_ts=?, current_price=?
                WHERE position_id=?
                """,
                (exit_price, exit_reason, time.time(), exit_price, position_id),
            )

    # ── HTTP ───────────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def _fetch_market_stats(self, market_id: str) -> dict[str, Any]:
        """Fetch current price + 24h volume from CLOB."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.clob_url}/markets/{market_id}"
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as exc:
            logger.debug("CLOB stats fetch failed for %s: %s", market_id, exc)
        return {}

    # ── Expected move ──────────────────────────────────────────────────────

    def calculate_expected_move(
        self, market: Any, patterns: dict[str, Any]
    ) -> float:
        """
        Estimate the expected price move for a market.

        For a YES token priced at p, the full move to resolution is (1 - p).
        We scale by the median_captured_pct from wallet patterns to get
        the realistic expected move.
        """
        yes_price: float = getattr(market, "yes_price", 0.5)
        full_move = 1.0 - yes_price
        captured_pct = float(patterns.get("median_captured_pct", 0.86))
        return full_move * captured_pct

    # ── Volume spike detection ─────────────────────────────────────────────

    async def detect_volume_spike(self, market_id: str) -> tuple[bool, float]:
        """
        Returns (spike_detected, current_ratio).
        A spike is current_volume > volume_spike_multiplier × baseline.
        """
        stats = await self._fetch_market_stats(market_id)
        current_vol = float(stats.get("volume24hr", stats.get("volume", 0)))

        history = self._volume_history.setdefault(market_id, [])
        now = time.time()
        history.append((now, current_vol))
        # Keep last 24h of observations
        cutoff = now - 86400
        self._volume_history[market_id] = [(t, v) for t, v in history if t >= cutoff]

        if len(history) < 3:
            return False, 1.0

        baseline = sum(v for _, v in history[:-1]) / (len(history) - 1)
        if baseline <= 0:
            return False, 1.0

        ratio = current_vol / baseline
        spike = ratio >= self.volume_spike_multiplier
        if spike:
            logger.info(
                "Volume spike detected for %s: %.1fx baseline", market_id, ratio
            )
        return spike, ratio

    # ── Signal generation ──────────────────────────────────────────────────

    def generate_signal(
        self,
        market: Any,
        patterns: dict[str, Any],
        score: float,
        claude_analysis: dict[str, Any] | None = None,
    ) -> TradeSignal | None:
        """
        Generate a trade signal from a scored market + wallet patterns.
        Returns None if no edge is found.
        """
        yes_price: float = getattr(market, "yes_price", 0.5)
        question: str = getattr(market, "question", "")
        market_id: str = getattr(market, "condition_id", "")

        # Minimum score to trade
        if score < 70:
            logger.debug("Market %s score %.1f too low", market_id, score)
            return None

        # Skip markets already in position
        if market_id in self._open_positions:
            return None

        # Price must be in tradeable range
        if not (0.10 <= yes_price <= 0.90):
            return None

        # Decide direction: YES if price < 0.5, NO if price > 0.5
        if yes_price <= 0.5:
            side = "YES"
            entry_price = yes_price
            expected_move = self.calculate_expected_move(market, patterns)
            target_price = min(entry_price + expected_move, 0.98)
            stop_price = entry_price * (1 - self.loss_cut)
        else:
            side = "NO"
            entry_price = 1.0 - yes_price
            full_move = 1.0 - entry_price
            captured_pct = float(patterns.get("median_captured_pct", 0.86))
            expected_move = full_move * captured_pct
            target_price = min(entry_price + expected_move, 0.98)
            stop_price = entry_price * (1 - self.loss_cut)

        # Confidence from score
        confidence = min(1.0, score / 100.0)

        # Incorporate Claude analysis if available
        rationale_parts = [
            f"Score: {score:.1f}/100",
            f"Direction: {side} @ {entry_price:.3f}",
            f"Target: {target_price:.3f} (+{expected_move*100:.1f}%)",
            f"Stop: {stop_price:.3f} (-{self.loss_cut*100:.1f}%)",
        ]
        if claude_analysis:
            sentiment = claude_analysis.get("sentiment", "neutral")
            edge = claude_analysis.get("edge_description", "")
            rationale_parts.append(f"Claude: {sentiment} — {edge}")

        return TradeSignal(
            market_id=market_id,
            question=question,
            side=side,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            confidence=confidence,
            rationale=" | ".join(rationale_parts),
            score=score,
        )

    # ── Exit logic ─────────────────────────────────────────────────────────

    async def should_exit(self, position: Position) -> tuple[bool, str]:
        """
        Returns (should_exit, reason).

        Exit conditions (whichever triggers first):
          1. Loss-cut: pnl_pct <= -loss_cut
          2. Target: captured >= exit_threshold of expected move
          3. Volume spike: current volume >= volume_spike_multiplier × baseline
        """
        # Loss cut
        if position.pnl_pct <= -self.loss_cut:
            return True, f"loss_cut ({position.pnl_pct*100:.1f}%)"

        # Profit target: how much of the move to target have we captured?
        if position.target_price > position.entry_price:
            move_to_target = position.target_price - position.entry_price
            captured = position.current_price - position.entry_price
            captured_fraction = captured / move_to_target if move_to_target > 0 else 0
            if captured_fraction >= self.exit_threshold:
                return True, (
                    f"target_captured ({captured_fraction*100:.1f}% of move)"
                )

        # Volume spike
        spike, ratio = await self.detect_volume_spike(position.market_id)
        if spike:
            return True, f"volume_spike ({ratio:.1f}x)"

        return False, ""

    # ── Position management ────────────────────────────────────────────────

    def open_position(self, signal: TradeSignal, filled_price: float, size_usdc: float) -> Position:
        """Record a newly filled position."""
        import uuid
        pos = Position(
            position_id=str(uuid.uuid4()),
            market_id=signal.market_id,
            question=signal.question,
            side=signal.side,
            entry_price=filled_price,
            current_price=filled_price,
            size_usdc=size_usdc,
            target_price=signal.target_price,
            stop_price=signal.stop_price,
        )
        self._open_positions[signal.market_id] = pos
        self._save_position(pos)
        logger.info(
            "Opened position: %s %s @ %.4f target=%.4f stop=%.4f",
            signal.side, signal.question[:50], filled_price,
            signal.target_price, signal.stop_price,
        )
        return pos

    def update_position_price(self, market_id: str, current_price: float) -> None:
        """Update mark-to-market price for an open position."""
        if market_id in self._open_positions:
            self._open_positions[market_id].current_price = current_price

    def close_position(self, market_id: str, exit_price: float, reason: str) -> Position | None:
        """Remove position from open set and record the close."""
        pos = self._open_positions.pop(market_id, None)
        if pos:
            self._close_position_db(pos.position_id, exit_price, reason)
            logger.info(
                "Closed position %s reason=%s pnl=%.2f%%",
                pos.question[:50], reason, pos.pnl_pct * 100,
            )
        return pos

    def get_open_positions(self) -> list[Position]:
        return list(self._open_positions.values())

    def get_portfolio_pnl(self) -> dict[str, float]:
        positions = self.get_open_positions()
        if not positions:
            return {"total_usdc": 0.0, "total_pct": 0.0, "position_count": 0}
        total_cost = sum(p.size_usdc for p in positions)
        total_pnl = sum(p.pnl_usdc for p in positions)
        return {
            "total_usdc": round(total_pnl, 2),
            "total_pct": round(total_pnl / total_cost * 100, 2) if total_cost else 0.0,
            "position_count": len(positions),
        }
