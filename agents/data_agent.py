"""
DataAgent — downloads the warproxxx/poly_data HuggingFace dataset,
analyses wallet performance, extracts behavioural patterns used by
top traders, and caches everything to SQLite.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataAgent:
    """Downloads and analyses the Polymarket 86M-trade dataset."""

    HF_DATASET = "warproxxx/poly_data"
    # Minimum trades a wallet must have to be considered
    MIN_TRADES = 50
    # Minimum win rate to qualify as a "top wallet"
    MIN_WIN_RATE = 0.60

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "wallet_analysis.db"
        self._init_db()

    # ── Database ───────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS wallet_stats (
                    wallet TEXT PRIMARY KEY,
                    total_trades INTEGER,
                    win_rate REAL,
                    avg_hold_seconds REAL,
                    early_exit_pct REAL,
                    avg_captured_pct REAL,
                    avg_loss_cut REAL,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS alpha_signals (
                    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_type TEXT,
                    description TEXT,
                    value REAL,
                    created_at TEXT
                );
                """
            )

    def _cache_wallet_stats(self, stats: list[dict[str, Any]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as con:
            con.executemany(
                """
                INSERT OR REPLACE INTO wallet_stats
                (wallet, total_trades, win_rate, avg_hold_seconds,
                 early_exit_pct, avg_captured_pct, avg_loss_cut, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s["wallet"],
                        s["total_trades"],
                        s["win_rate"],
                        s["avg_hold_seconds"],
                        s["early_exit_pct"],
                        s["avg_captured_pct"],
                        s["avg_loss_cut"],
                        now,
                    )
                    for s in stats
                ],
            )

    def _load_cached_wallets(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM wallet_stats ORDER BY win_rate DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Core analysis ──────────────────────────────────────────────────────

    def _load_dataset(self) -> pd.DataFrame:
        """Stream-load the HuggingFace dataset into a DataFrame."""
        try:
            from datasets import load_dataset  # type: ignore

            logger.info("Loading %s from HuggingFace …", self.HF_DATASET)
            ds = load_dataset(self.HF_DATASET, split="train", streaming=False)
            df = ds.to_pandas()
            logger.info("Loaded %d rows", len(df))
            return df
        except Exception as exc:
            logger.error("Failed to load dataset: %s", exc)
            raise

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        poly_data column names vary by version.  We normalise to a
        consistent schema so the rest of the code is stable.

        Expected raw columns (best-effort):
            wallet / user_address
            market_id / conditionId
            side (BUY / SELL)
            price
            size / amount
            timestamp / created_at
            outcome (WIN / LOSS / OPEN)
            resolution_timestamp (optional)
        """
        rename_map: dict[str, str] = {}
        lower = {c.lower(): c for c in df.columns}

        def _pick(candidates: list[str], target: str) -> None:
            for c in candidates:
                if c in lower:
                    rename_map[lower[c]] = target
                    return

        _pick(["wallet", "user_address", "trader"], "wallet")
        _pick(["market_id", "conditionid", "condition_id"], "market_id")
        _pick(["side", "type", "trade_type"], "side")
        _pick(["price", "avg_price", "fill_price"], "price")
        _pick(["size", "amount", "qty", "quantity"], "size")
        _pick(["timestamp", "created_at", "time"], "timestamp")
        _pick(["outcome", "result", "pnl_sign"], "outcome")
        _pick(["resolution_timestamp", "resolution_time", "expiry"], "resolution_ts")

        df = df.rename(columns=rename_map)

        # Ensure numeric types
        for col in ("price", "size"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if "resolution_ts" in df.columns:
            df["resolution_ts"] = pd.to_datetime(
                df["resolution_ts"], utc=True, errors="coerce"
            )

        return df

    def _compute_wallet_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each wallet compute:
            - total_trades
            - win_rate
            - avg_hold_seconds
            - early_exit_pct   (% of trades exited before resolution)
            - avg_captured_pct (average % of total move captured)
            - avg_loss_cut     (average loss on losing trades)
        """
        required = {"wallet", "market_id", "price", "size", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing expected columns: {missing}")

        results: list[dict[str, Any]] = []

        # Group by wallet
        for wallet, wdf in df.groupby("wallet"):
            if len(wdf) < self.MIN_TRADES:
                continue

            # Separate buys and sells
            buys = wdf[wdf.get("side", pd.Series(dtype=str)).str.upper().isin(["BUY", "B"])]
            sells = wdf[wdf.get("side", pd.Series(dtype=str)).str.upper().isin(["SELL", "S"])]

            # Win rate — use outcome column if present
            if "outcome" in df.columns:
                outcomes = wdf["outcome"].str.upper()
                wins = (outcomes == "WIN").sum()
                losses = (outcomes == "LOSS").sum()
                total = wins + losses
                win_rate = float(wins / total) if total > 0 else 0.0
            else:
                # Approximate: sell price > buy price for same market
                win_rate = self._approximate_win_rate(wdf)

            # Hold time: for each market where this wallet has both a buy
            # and a sell, compute sell_ts - buy_ts
            hold_times: list[float] = []
            early_exits: list[bool] = []
            captured_pcts: list[float] = []

            for market_id, mdf in wdf.groupby("market_id"):
                m_buys = mdf[mdf.get("side", pd.Series(dtype=str)).str.upper().isin(["BUY", "B"])]
                m_sells = mdf[mdf.get("side", pd.Series(dtype=str)).str.upper().isin(["SELL", "S"])]
                if m_buys.empty or m_sells.empty:
                    continue
                entry_ts = m_buys["timestamp"].min()
                exit_ts = m_sells["timestamp"].max()
                hold_sec = (exit_ts - entry_ts).total_seconds()
                if hold_sec > 0:
                    hold_times.append(hold_sec)

                # Early exit: sold before resolution?
                if "resolution_ts" in mdf.columns:
                    res_ts = mdf["resolution_ts"].dropna()
                    if not res_ts.empty:
                        resolution = res_ts.iloc[0]
                        early_exits.append(exit_ts < resolution)

                # Captured pct: (sell_price - buy_price) / (1 - buy_price)
                buy_px = float(m_buys["price"].mean())
                sell_px = float(m_sells["price"].mean())
                if 0 < buy_px < 1:
                    full_move = 1.0 - buy_px
                    if full_move > 0:
                        captured = (sell_px - buy_px) / full_move
                        captured_pcts.append(min(captured, 1.0))

            # Loss-cut stats from losing trades
            loss_cuts: list[float] = []
            if "outcome" in df.columns:
                losing = wdf[wdf["outcome"].str.upper() == "LOSS"]
                for _, row in losing.iterrows():
                    # Loss % approximated from price if we have entry/exit
                    if "price" in row and row["price"] < 1:
                        loss_cuts.append(1.0 - float(row["price"]))

            stats: dict[str, Any] = {
                "wallet": wallet,
                "total_trades": len(wdf),
                "win_rate": win_rate,
                "avg_hold_seconds": float(np.mean(hold_times)) if hold_times else 0.0,
                "early_exit_pct": float(np.mean(early_exits)) if early_exits else 0.0,
                "avg_captured_pct": float(np.mean(captured_pcts)) if captured_pcts else 0.0,
                "avg_loss_cut": float(np.mean(loss_cuts)) if loss_cuts else 0.0,
            }
            results.append(stats)

        return pd.DataFrame(results)

    @staticmethod
    def _approximate_win_rate(wdf: pd.DataFrame) -> float:
        """Naive win-rate: fraction of markets where avg sell > avg buy."""
        wins = 0
        total = 0
        side_col = wdf.get("side", pd.Series(dtype=str)).str.upper()
        for _, mdf in wdf.groupby("market_id"):
            m_side = side_col.loc[mdf.index]
            buys = mdf[m_side.isin(["BUY", "B"])]
            sells = mdf[m_side.isin(["SELL", "S"])]
            if buys.empty or sells.empty:
                continue
            buy_px = buys["price"].mean()
            sell_px = sells["price"].mean()
            total += 1
            if sell_px > buy_px:
                wins += 1
        return float(wins / total) if total > 0 else 0.5

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze_top_wallets(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        """
        Run full wallet analysis and return top wallets sorted by win rate.
        Results are cached in SQLite; pass force_refresh=True to re-compute.
        """
        cached = self._load_cached_wallets()
        if cached and not force_refresh:
            logger.info("Returning %d cached wallet records", len(cached))
            return cached

        logger.info("Starting full wallet analysis …")
        df_raw = self._load_dataset()
        df = self._normalise_columns(df_raw)
        stats_df = self._compute_wallet_stats(df)

        # Filter to top wallets
        top = stats_df[
            (stats_df["win_rate"] >= self.MIN_WIN_RATE)
            & (stats_df["total_trades"] >= self.MIN_TRADES)
        ].sort_values("win_rate", ascending=False)

        records = top.to_dict(orient="records")
        self._cache_wallet_stats(records)
        logger.info("Identified %d top wallets", len(records))
        return records

    def get_wallet_patterns(self) -> dict[str, Any]:
        """
        Aggregate cross-wallet behavioural patterns from cached data.
        Returns a dict with median/mean metrics ready for StrategyAgent.
        """
        wallets = self._load_cached_wallets()
        if not wallets:
            logger.warning("No cached wallet data — run analyze_top_wallets() first")
            return {}

        df = pd.DataFrame(wallets)
        patterns: dict[str, Any] = {
            "wallet_count": len(df),
            "median_win_rate": float(df["win_rate"].median()),
            "median_hold_hours": float(df["avg_hold_seconds"].median() / 3600),
            "median_early_exit_pct": float(df["early_exit_pct"].median()),
            "median_captured_pct": float(df["avg_captured_pct"].median()),
            "median_loss_cut": float(df["avg_loss_cut"].median()),
            # Distribution percentiles
            "p25_captured_pct": float(df["avg_captured_pct"].quantile(0.25)),
            "p75_captured_pct": float(df["avg_captured_pct"].quantile(0.75)),
            "p25_hold_hours": float((df["avg_hold_seconds"] / 3600).quantile(0.25)),
            "p75_hold_hours": float((df["avg_hold_seconds"] / 3600).quantile(0.75)),
        }
        logger.info("Wallet patterns: %s", patterns)
        return patterns

    def find_alpha_signals(self) -> list[dict[str, Any]]:
        """
        Derive actionable alpha signals from wallet patterns.
        Each signal has a type, description, and numeric value.
        """
        patterns = self.get_wallet_patterns()
        if not patterns:
            return []

        signals: list[dict[str, Any]] = [
            {
                "signal_type": "early_exit",
                "description": (
                    f"{patterns['median_early_exit_pct']*100:.1f}% of top-wallet "
                    "trades exit before resolution — don't hold to expiry"
                ),
                "value": patterns["median_early_exit_pct"],
            },
            {
                "signal_type": "capture_target",
                "description": (
                    f"Top wallets capture a median "
                    f"{patterns['median_captured_pct']*100:.1f}% of the expected move"
                ),
                "value": patterns["median_captured_pct"],
            },
            {
                "signal_type": "loss_cut",
                "description": (
                    f"Median loss-cut at "
                    f"{patterns['median_loss_cut']*100:.1f}% of position value"
                ),
                "value": patterns["median_loss_cut"],
            },
            {
                "signal_type": "hold_time",
                "description": (
                    f"Median hold time "
                    f"{patterns['median_hold_hours']:.1f}h "
                    f"(P25={patterns['p25_hold_hours']:.1f}h, "
                    f"P75={patterns['p75_hold_hours']:.1f}h)"
                ),
                "value": patterns["median_hold_hours"],
            },
        ]

        # Persist signals
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM alpha_signals")
            con.executemany(
                "INSERT INTO alpha_signals (signal_type, description, value, created_at) "
                "VALUES (?, ?, ?, ?)",
                [(s["signal_type"], s["description"], s["value"], now) for s in signals],
            )
        return signals
