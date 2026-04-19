"""
Backtester for the Polymarket exit strategy.

Loads historical trade data from poly_data, simulates the
85%-capture / 3×-volume-spike strategy, and compares against
holding to resolution.

Usage:
    python scripts/backtest.py [--sample-size 10000] [--output results.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from agents.data_agent import DataAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("backtest")


# ── Trade simulation ───────────────────────────────────────────────────────

@dataclass
class SimTrade:
    market_id: str
    wallet: str
    entry_price: float
    exit_price_strategy: float   # exit via our rules
    exit_price_hold: float       # exit at resolution
    hold_seconds: float
    strategy_exit_reason: str
    won_strategy: bool
    won_hold: bool
    pnl_pct_strategy: float
    pnl_pct_hold: float


@dataclass
class BacktestResults:
    strategy_name: str
    total_trades: int
    win_rate: float
    avg_pnl_pct: float
    total_return_pct: float
    max_drawdown_pct: float
    avg_hold_hours: float
    sharpe_ratio: float
    exit_reason_counts: dict[str, int] = field(default_factory=dict)


def _simulate_trade(
    row: dict[str, Any],
    exit_threshold: float,
    loss_cut: float,
    volume_spike_mult: float,
) -> SimTrade | None:
    """
    Simulate one round-trip trade using our exit strategy.

    We approximate:
      - Entry price = buy price
      - Max achievable price ≈ 0.98 (binary market cap)
      - Volume spike: flagged if hold > median_hold × 3 (proxy for the spike)
      - Strategy exit: first of (threshold hit, loss cut, spike proxy, resolution)
      - Hold exit: resolution price (WIN → 1.0, LOSS → 0.0)
    """
    entry = float(row.get("entry_price", row.get("price", 0.5)))
    resolution_price = float(row.get("resolution_price", 1.0 if row.get("outcome", "").upper() == "WIN" else 0.0))
    hold_sec = float(row.get("hold_seconds", 86400))

    if entry <= 0 or entry >= 1:
        return None

    full_move = 1.0 - entry
    if full_move <= 0:
        return None

    # Simulate price path: linear drift toward resolution then snap
    target_price = entry + full_move * exit_threshold
    stop_price = entry * (1 - loss_cut)

    # Determine strategy exit
    strategy_exit_price = entry
    exit_reason = "hold_to_resolution"

    # Loss cut
    if resolution_price < stop_price:
        strategy_exit_price = stop_price
        exit_reason = "loss_cut"
    # Volume spike proxy: if hold is > 3× median (4h), assume spike fires
    elif hold_sec > 3 * 4 * 3600:
        # Conservative: exit at 70% of move
        strategy_exit_price = entry + full_move * 0.70
        exit_reason = "volume_spike"
    # Target reached
    elif resolution_price >= target_price:
        strategy_exit_price = target_price
        exit_reason = "target_captured"
    # Resolution without hitting targets
    else:
        strategy_exit_price = resolution_price
        exit_reason = "hold_to_resolution"

    pnl_strategy = (strategy_exit_price - entry) / entry
    pnl_hold = (resolution_price - entry) / entry

    return SimTrade(
        market_id=str(row.get("market_id", "")),
        wallet=str(row.get("wallet", "")),
        entry_price=entry,
        exit_price_strategy=strategy_exit_price,
        exit_price_hold=resolution_price,
        hold_seconds=hold_sec,
        strategy_exit_reason=exit_reason,
        won_strategy=pnl_strategy > 0,
        won_hold=pnl_hold > 0,
        pnl_pct_strategy=pnl_strategy,
        pnl_pct_hold=pnl_hold,
    )


def _compute_max_drawdown(returns: list[float]) -> float:
    """Compute maximum drawdown from a list of cumulative returns."""
    if not returns:
        return 0.0
    cumulative = np.cumprod([1 + r for r in returns])
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    return float(np.min(drawdowns))


def _compute_sharpe(returns: list[float], risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    excess = arr - risk_free
    std = np.std(excess)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def _compute_results(trades: list[SimTrade], strategy: bool) -> BacktestResults:
    pnls = [t.pnl_pct_strategy if strategy else t.pnl_pct_hold for t in trades]
    wins = [t.won_strategy if strategy else t.won_hold for t in trades]
    hold_hours = [t.hold_seconds / 3600 for t in trades]
    reason_counts: dict[str, int] = {}
    if strategy:
        for t in trades:
            reason_counts[t.strategy_exit_reason] = reason_counts.get(t.strategy_exit_reason, 0) + 1

    total_return = float(np.prod([1 + p for p in pnls]) - 1) if pnls else 0.0
    max_dd = _compute_max_drawdown(pnls)

    return BacktestResults(
        strategy_name="85%_exit_strategy" if strategy else "hold_to_resolution",
        total_trades=len(trades),
        win_rate=round(sum(wins) / len(wins), 4) if wins else 0.0,
        avg_pnl_pct=round(float(np.mean(pnls)), 4) if pnls else 0.0,
        total_return_pct=round(total_return * 100, 2),
        max_drawdown_pct=round(max_dd * 100, 2),
        avg_hold_hours=round(float(np.mean(hold_hours)), 2) if hold_hours else 0.0,
        sharpe_ratio=round(_compute_sharpe(pnls), 3),
        exit_reason_counts=reason_counts,
    )


# ── Data loading ───────────────────────────────────────────────────────────

def _load_backtest_data(sample_size: int) -> pd.DataFrame:
    """
    Load poly_data and prepare it for backtesting.
    We need completed round-trips: a buy + a sell in the same market.
    """
    logger.info("Loading poly_data from HuggingFace …")
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(DataAgent.HF_DATASET, split="train", streaming=False)
    df = ds.to_pandas()
    logger.info("Loaded %d raw rows", len(df))

    # Normalise columns
    agent = DataAgent(data_dir=settings.data_dir)
    df = agent._normalise_columns(df)

    required = {"wallet", "market_id", "price", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Build round-trips
    side_col = df.get("side", pd.Series("BUY", index=df.index)).str.upper()
    trips: list[dict[str, Any]] = []

    for (wallet, market_id), group in df.groupby(["wallet", "market_id"]):
        g_side = side_col.loc[group.index]
        buys = group[g_side.isin(["BUY", "B"])]
        sells = group[g_side.isin(["SELL", "S"])]
        if buys.empty or sells.empty:
            continue

        entry_price = float(buys["price"].mean())
        exit_price = float(sells["price"].mean())
        entry_ts = buys["timestamp"].min()
        exit_ts = sells["timestamp"].max()
        hold_sec = (exit_ts - entry_ts).total_seconds()

        outcome = "WIN" if exit_price > entry_price else "LOSS"
        if "outcome" in group.columns and not group["outcome"].isna().all():
            outcome = str(group["outcome"].dropna().iloc[-1]).upper()

        resolution_price = 1.0 if outcome == "WIN" else 0.0

        trips.append({
            "wallet": wallet,
            "market_id": market_id,
            "entry_price": entry_price,
            "resolution_price": resolution_price,
            "hold_seconds": max(hold_sec, 0),
            "outcome": outcome,
        })

        if len(trips) >= sample_size:
            break

    logger.info("Built %d round-trip trades for backtest", len(trips))
    return pd.DataFrame(trips)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Polymarket exit strategy")
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--output", type=str, default="data/backtest_results.json")
    parser.add_argument(
        "--exit-threshold", type=float, default=settings.exit_threshold
    )
    parser.add_argument("--loss-cut", type=float, default=settings.loss_cut)
    parser.add_argument(
        "--volume-spike-mult", type=float, default=settings.volume_spike_multiplier
    )
    args = parser.parse_args()

    try:
        df = _load_backtest_data(args.sample_size)
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        sys.exit(1)

    logger.info("Simulating %d trades …", len(df))
    sim_trades: list[SimTrade] = []
    for _, row in df.iterrows():
        t = _simulate_trade(
            row.to_dict(),
            exit_threshold=args.exit_threshold,
            loss_cut=args.loss_cut,
            volume_spike_mult=args.volume_spike_mult,
        )
        if t:
            sim_trades.append(t)

    if not sim_trades:
        logger.error("No valid trades simulated")
        sys.exit(1)

    strategy_res = _compute_results(sim_trades, strategy=True)
    hold_res = _compute_results(sim_trades, strategy=False)

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BACKTEST RESULTS")
    print("=" * 65)

    def _print_results(r: BacktestResults) -> None:
        print(f"\n  Strategy: {r.strategy_name}")
        print(f"    Trades:         {r.total_trades:,}")
        print(f"    Win rate:       {r.win_rate*100:.1f}%")
        print(f"    Avg PnL/trade:  {r.avg_pnl_pct*100:+.2f}%")
        print(f"    Total return:   {r.total_return_pct:+.1f}%")
        print(f"    Max drawdown:   {r.max_drawdown_pct:.1f}%")
        print(f"    Avg hold:       {r.avg_hold_hours:.1f}h")
        print(f"    Sharpe ratio:   {r.sharpe_ratio:.2f}")
        if r.exit_reason_counts:
            print(f"    Exit reasons:")
            for reason, count in sorted(r.exit_reason_counts.items(), key=lambda x: -x[1]):
                pct = count / r.total_trades * 100
                print(f"      {reason:<30} {count:>6,}  ({pct:.1f}%)")

    _print_results(strategy_res)
    _print_results(hold_res)

    improvement = strategy_res.win_rate - hold_res.win_rate
    print(f"\n  Win rate improvement vs hold-to-resolution: {improvement*100:+.1f}pp")
    print("=" * 65 + "\n")

    # ── Save JSON ──────────────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "exit_threshold": args.exit_threshold,
            "loss_cut": args.loss_cut,
            "volume_spike_multiplier": args.volume_spike_mult,
            "sample_size": len(sim_trades),
        },
        "strategy_85pct_exit": {
            "win_rate": strategy_res.win_rate,
            "avg_pnl_pct": strategy_res.avg_pnl_pct,
            "total_return_pct": strategy_res.total_return_pct,
            "max_drawdown_pct": strategy_res.max_drawdown_pct,
            "avg_hold_hours": strategy_res.avg_hold_hours,
            "sharpe_ratio": strategy_res.sharpe_ratio,
            "exit_reason_counts": strategy_res.exit_reason_counts,
        },
        "hold_to_resolution_baseline": {
            "win_rate": hold_res.win_rate,
            "avg_pnl_pct": hold_res.avg_pnl_pct,
            "total_return_pct": hold_res.total_return_pct,
            "max_drawdown_pct": hold_res.max_drawdown_pct,
            "avg_hold_hours": hold_res.avg_hold_hours,
            "sharpe_ratio": hold_res.sharpe_ratio,
        },
        "improvement_vs_baseline_pp": round(improvement * 100, 2),
    }

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
