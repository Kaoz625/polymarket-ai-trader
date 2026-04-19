"""
Standalone wallet-analysis script.

Downloads warproxxx/poly_data from HuggingFace, analyses all wallets,
and writes a detailed report to data/wallet_analysis.json.

Usage:
    python scripts/analyze_poly_data.py [--force-refresh]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path when running as a script
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from agents.data_agent import DataAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("analyze_poly_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse warproxxx/poly_data wallet behaviour")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download and re-compute even if cache exists",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top wallets to include in the output (default: 100)",
    )
    args = parser.parse_args()

    agent = DataAgent(data_dir=settings.data_dir)

    # ── Step 1: Wallet stats ───────────────────────────────────────────────
    logger.info("Step 1/3 — Analysing top wallets …")
    try:
        top_wallets = agent.analyze_top_wallets(force_refresh=args.force_refresh)
    except Exception as exc:
        logger.error("Wallet analysis failed: %s", exc)
        sys.exit(1)

    logger.info("Found %d top-performing wallets", len(top_wallets))

    # ── Step 2: Patterns ───────────────────────────────────────────────────
    logger.info("Step 2/3 — Extracting behavioural patterns …")
    patterns = agent.get_wallet_patterns()

    # ── Step 3: Alpha signals ──────────────────────────────────────────────
    logger.info("Step 3/3 — Deriving alpha signals …")
    signals = agent.find_alpha_signals()

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  POLYMARKET WALLET ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Top wallets analysed : {len(top_wallets)}")
    print(f"  Median win rate      : {patterns.get('median_win_rate', 0)*100:.1f}%")
    print(f"  Median hold time     : {patterns.get('median_hold_hours', 0):.1f} hours")
    print(f"  Early-exit rate      : {patterns.get('median_early_exit_pct', 0)*100:.1f}%")
    print(f"  Avg move captured    : {patterns.get('median_captured_pct', 0)*100:.1f}%")
    print(f"  Median loss cut      : {patterns.get('median_loss_cut', 0)*100:.1f}%")
    print()

    print("  ALPHA SIGNALS")
    print("-" * 60)
    for s in signals:
        print(f"  [{s['signal_type'].upper()}] {s['description']}")
    print()

    print("  TOP 10 WALLETS BY WIN RATE")
    print("-" * 60)
    for w in top_wallets[:10]:
        print(
            f"  {str(w.get('wallet', ''))[:12]}…  "
            f"win={w.get('win_rate', 0)*100:.1f}%  "
            f"trades={w.get('total_trades', 0)}  "
            f"early_exit={w.get('early_exit_pct', 0)*100:.1f}%  "
            f"captured={w.get('avg_captured_pct', 0)*100:.1f}%"
        )
    print("=" * 60 + "\n")

    # ── Save JSON ──────────────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DataAgent.HF_DATASET,
        "top_wallet_count": len(top_wallets),
        "patterns": patterns,
        "signals": signals,
        "exit_timing_stats": {
            "pct_exit_before_resolution": patterns.get("median_early_exit_pct", 0.91),
            "median_pct_move_captured": patterns.get("median_captured_pct", 0.86),
            "median_loss_cut_pct": patterns.get("median_loss_cut", 0.12),
            "recommended_exit_threshold": settings.exit_threshold,
            "recommended_loss_cut": settings.loss_cut,
        },
        "top_wallets": top_wallets[: args.top_n],
    }

    out_path = settings.wallet_analysis_path
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    logger.info("Analysis saved to %s", out_path)
    print(f"Output written to: {out_path}\n")


if __name__ == "__main__":
    main()
