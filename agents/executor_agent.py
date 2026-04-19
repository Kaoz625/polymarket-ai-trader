"""
ExecutorAgent — connects to Polymarket CLOB API via py-clob-client,
places and manages orders, logs all trades to SQLite.
"""
from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    order_id: str
    market_id: str
    side: str
    size: float
    price: float
    status: str
    filled_price: float = 0.0
    filled_size: float = 0.0
    error: str = ""
    timestamp: float = field(default_factory=time.time)


class ExecutorAgent:
    """
    Places orders on Polymarket CLOB and tracks positions in SQLite.

    In dry-run mode no real orders are sent — all operations are simulated
    at the current market price.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        private_key: str,
        db_path: str | Path,
        max_position_usdc: float = 100.0,
        dry_run: bool = False,
        clob_url: str = "https://clob.polymarket.com",
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.private_key = private_key
        self.db_path = str(db_path)
        self.max_position_usdc = max_position_usdc
        self.dry_run = dry_run
        self.clob_url = clob_url
        self._client: Any = None
        self._init_db()

    # ── Database ───────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    order_id TEXT,
                    market_id TEXT,
                    question TEXT,
                    side TEXT,
                    size_usdc REAL,
                    requested_price REAL,
                    filled_price REAL,
                    filled_size REAL,
                    status TEXT,
                    error TEXT,
                    dry_run INTEGER,
                    created_at REAL
                );
                CREATE TABLE IF NOT EXISTS open_positions (
                    position_id TEXT PRIMARY KEY,
                    market_id TEXT,
                    question TEXT,
                    side TEXT,
                    entry_price REAL,
                    size_usdc REAL,
                    entry_order_id TEXT,
                    opened_at REAL
                );
                CREATE TABLE IF NOT EXISTS closed_positions (
                    position_id TEXT PRIMARY KEY,
                    market_id TEXT,
                    question TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    size_usdc REAL,
                    pnl_usdc REAL,
                    pnl_pct REAL,
                    exit_reason TEXT,
                    opened_at REAL,
                    closed_at REAL
                );
                """
            )

    def _log_trade(self, result: OrderResult, question: str = "", dry: bool = False) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                INSERT OR REPLACE INTO trades
                (trade_id, order_id, market_id, question, side, size_usdc,
                 requested_price, filled_price, filled_size, status, error, dry_run, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()), result.order_id, result.market_id,
                    question, result.side, result.size, result.price,
                    result.filled_price, result.filled_size, result.status,
                    result.error, int(dry), result.timestamp,
                ),
            )

    # ── CLOB client ────────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        """Lazily initialise the py-clob-client."""
        if self._client is not None:
            return self._client
        try:
            from py_clob_client.client import ClobClient  # type: ignore
            from py_clob_client.clob_types import ApiCreds  # type: ignore

            creds = ApiCreds(
                api_key=self.api_key,
                api_secret=self.api_secret,
                api_passphrase=self.api_passphrase,
            )
            self._client = ClobClient(
                host=self.clob_url,
                key=self.private_key,
                chain_id=137,  # Polygon mainnet
                creds=creds,
            )
            logger.info("CLOB client initialised")
        except ImportError:
            logger.error("py-clob-client not installed — run: pip install py-clob-client")
            raise
        except Exception as exc:
            logger.error("CLOB client init failed: %s", exc)
            raise
        return self._client

    # ── Balance & positions ────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Return USDC balance. Returns 0.0 in dry-run mode."""
        if self.dry_run:
            logger.info("[DRY RUN] get_balance() → 10000.00 USDC (simulated)")
            return 10_000.0
        try:
            client = self._get_client()
            balance = client.get_balance()
            usdc = float(balance.get("USDC", balance) if isinstance(balance, dict) else balance)
            logger.info("Balance: %.2f USDC", usdc)
            return usdc
        except Exception as exc:
            logger.error("get_balance failed: %s", exc)
            return 0.0

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Return open positions from SQLite."""
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM open_positions ORDER BY opened_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_closed_positions(self, limit: int = 100) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM closed_positions ORDER BY closed_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def compute_win_rate(self) -> float:
        """Calculate win rate from closed positions."""
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT COUNT(*) total, SUM(CASE WHEN pnl_usdc > 0 THEN 1 ELSE 0 END) wins "
                "FROM closed_positions"
            ).fetchone()
        total, wins = row
        if not total:
            return 0.0
        return round(wins / total, 4)

    # ── Order placement ────────────────────────────────────────────────────

    def place_order(
        self,
        signal: Any,
        size_usdc: float | None = None,
    ) -> OrderResult:
        """
        Place a limit order for a TradeSignal.
        size_usdc defaults to max_position_usdc.
        """
        trade_size = min(size_usdc or self.max_position_usdc, self.max_position_usdc)
        market_id = signal.market_id
        side = signal.side
        price = signal.entry_price
        question = getattr(signal, "question", "")

        if self.dry_run:
            result = OrderResult(
                order_id=f"dry_{uuid.uuid4().hex[:8]}",
                market_id=market_id,
                side=side,
                size=trade_size,
                price=price,
                status="filled",
                filled_price=price,
                filled_size=trade_size / price,
            )
            logger.info(
                "[DRY RUN] Place %s %s @ %.4f size=%.2f USDC",
                side, question[:50], price, trade_size,
            )
            self._log_trade(result, question, dry=True)
            self._record_open_position(result, question)
            return result

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType  # type: ignore

            client = self._get_client()
            # Convert USDC size to token units
            token_size = trade_size / price if price > 0 else trade_size

            order_args = OrderArgs(
                token_id=market_id,
                price=price,
                size=round(token_size, 4),
                side=side,
            )
            resp = client.create_and_post_order(order_args)
            order_id = resp.get("orderID", resp.get("id", str(uuid.uuid4())))
            filled_price = float(resp.get("price", price))
            filled_size = float(resp.get("sizeMatched", token_size))
            status = resp.get("status", "LIVE")

            result = OrderResult(
                order_id=order_id,
                market_id=market_id,
                side=side,
                size=trade_size,
                price=price,
                status=status,
                filled_price=filled_price,
                filled_size=filled_size,
            )
            logger.info(
                "Order placed: %s %s @ %.4f id=%s status=%s",
                side, question[:50], price, order_id, status,
            )
            self._log_trade(result, question, dry=False)
            if status in ("MATCHED", "filled"):
                self._record_open_position(result, question)
            return result

        except Exception as exc:
            logger.error("place_order failed for %s: %s", market_id, exc)
            result = OrderResult(
                order_id="",
                market_id=market_id,
                side=side,
                size=trade_size,
                price=price,
                status="error",
                error=str(exc),
            )
            self._log_trade(result, question, dry=False)
            return result

    def _record_open_position(self, result: OrderResult, question: str) -> None:
        pos_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                """
                INSERT INTO open_positions
                (position_id, market_id, question, side, entry_price, size_usdc,
                 entry_order_id, opened_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pos_id, result.market_id, question, result.side,
                    result.filled_price or result.price, result.size,
                    result.order_id, result.timestamp,
                ),
            )

    # ── Consensus voting ───────────────────────────────────────────────────

    def consensus_execute(
        self,
        agents: list[Any],
        market: Any,
        wallet_balance: float,
        signal: Any | None = None,
    ) -> "OrderResult | None":
        """
        Collect synchronous evaluate() votes from a list of agents and execute
        with Kelly sizing based on consensus strength.

        Voting rules:
            2+ agents return action="BUY" → full Kelly position
            1 agent returns action="BUY"  → half Kelly position
            0 agents return action="BUY"  → no trade

        Args:
            agents:         List of agent objects with an evaluate(market) method.
                            evaluate() must return {"action": str, "confidence": float, ...}.
                            Async agents are NOT supported here — use executor_process.py
                            for async consensus.
            market:         Market object (must have .yes_price and .condition_id).
            wallet_balance: Current USDC balance for Kelly sizing.
            signal:         Pre-built signal object.  If None, a minimal signal is
                            constructed from market attributes.

        Returns:
            OrderResult if an order was placed, None otherwise.
        """
        from agents.kelly import kelly_size  # local import avoids circular deps

        # Minimum volume filter
        volume_24h = float(getattr(market, "volume_24h", 0))
        if volume_24h < 50_000:
            logger.info(
                "consensus_execute: SKIP volume %.0f < 50000 for %s",
                volume_24h, getattr(market, "condition_id", "?")[:20],
            )
            return None

        # Collect votes
        buy_votes = 0
        total_confidence = 0.0
        for agent in agents:
            try:
                result = agent.evaluate(market)
                # Handle coroutines returned by async agents (best-effort)
                import asyncio
                import inspect
                if inspect.iscoroutine(result):
                    try:
                        result = asyncio.get_event_loop().run_until_complete(result)
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        result = loop.run_until_complete(result)
                        loop.close()
                action = result.get("action", "PASS")
                conf = float(result.get("confidence", 0.0))
                if action == "BUY":
                    buy_votes += 1
                    total_confidence += conf
            except Exception as exc:
                logger.warning("Agent %s evaluate failed: %s", type(agent).__name__, exc)

        logger.info(
            "consensus_execute: %d/%d BUY votes for %s",
            buy_votes, len(agents), getattr(market, "condition_id", "?")[:20],
        )

        if buy_votes == 0:
            return None

        # Build a minimal signal if none provided
        if signal is None:
            yes_price = float(getattr(market, "yes_price", 0.5))
            loss_cut = 0.12

            class _MinimalSignal:
                market_id = getattr(market, "condition_id", "")
                question = getattr(market, "question", "")
                side = "YES"
                entry_price = yes_price
                target_price = min(yes_price + (1.0 - yes_price) * 0.86, 0.98)
                stop_price = yes_price * (1 - loss_cut)

            signal = _MinimalSignal()

        # Kelly sizing
        avg_confidence = total_confidence / max(buy_votes, 1)
        full_kelly = kelly_size(
            p_win=avg_confidence,
            market_price=signal.entry_price,
            bankroll=wallet_balance,
            max_fraction=0.25,
        )

        if buy_votes >= 2:
            size_usdc = full_kelly
        else:
            size_usdc = full_kelly * 0.5

        size_usdc = min(size_usdc, self.max_position_usdc)

        if size_usdc <= 0:
            logger.info("consensus_execute: Kelly returned 0, skipping")
            return None

        return self.place_order(signal, size_usdc=size_usdc)

    # ── Position closing ───────────────────────────────────────────────────

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str = "manual",
    ) -> OrderResult | None:
        """
        Close an open position by placing a market-sell order.
        """
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                "SELECT * FROM open_positions WHERE position_id=?", (position_id,)
            ).fetchone()

        if not row:
            logger.warning("Position %s not found", position_id)
            return None

        pos = dict(row)
        market_id = pos["market_id"]
        entry_price = pos["entry_price"]
        size_usdc = pos["size_usdc"]
        side = pos["side"]
        question = pos["question"]

        # Close side is opposite
        close_side = "SELL" if side == "YES" else "BUY"
        pnl_pct = (exit_price - entry_price) / entry_price if side == "YES" \
            else (entry_price - exit_price) / entry_price
        pnl_usdc = pnl_pct * size_usdc

        if self.dry_run:
            close_result = OrderResult(
                order_id=f"dry_close_{uuid.uuid4().hex[:8]}",
                market_id=market_id,
                side=close_side,
                size=size_usdc,
                price=exit_price,
                status="filled",
                filled_price=exit_price,
                filled_size=size_usdc / exit_price if exit_price else 0,
            )
            logger.info(
                "[DRY RUN] Close %s @ %.4f reason=%s pnl=%.2f%%",
                question[:50], exit_price, exit_reason, pnl_pct * 100,
            )
        else:
            try:
                from py_clob_client.clob_types import OrderArgs  # type: ignore
                client = self._get_client()
                token_size = size_usdc / exit_price if exit_price else 0
                order_args = OrderArgs(
                    token_id=market_id,
                    price=exit_price,
                    size=round(token_size, 4),
                    side=close_side,
                )
                resp = client.create_and_post_order(order_args)
                order_id = resp.get("orderID", str(uuid.uuid4()))
                filled_price = float(resp.get("price", exit_price))
                exit_price = filled_price  # use actual fill
                pnl_pct = (exit_price - entry_price) / entry_price if side == "YES" \
                    else (entry_price - exit_price) / entry_price
                pnl_usdc = pnl_pct * size_usdc

                close_result = OrderResult(
                    order_id=order_id,
                    market_id=market_id,
                    side=close_side,
                    size=size_usdc,
                    price=exit_price,
                    status="filled",
                    filled_price=exit_price,
                )
            except Exception as exc:
                logger.error("close_position failed: %s", exc)
                return None

        # Move from open → closed in DB
        now = time.time()
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM open_positions WHERE position_id=?", (position_id,))
            con.execute(
                """
                INSERT INTO closed_positions
                (position_id, market_id, question, side, entry_price, exit_price,
                 size_usdc, pnl_usdc, pnl_pct, exit_reason, opened_at, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position_id, market_id, question, side, entry_price,
                    exit_price, size_usdc, round(pnl_usdc, 4),
                    round(pnl_pct, 6), exit_reason,
                    pos["opened_at"], now,
                ),
            )

        self._log_trade(close_result, question, dry=self.dry_run)
        return close_result
