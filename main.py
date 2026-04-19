"""
Polymarket AI Trading System — main orchestrator.

Usage:
    python main.py                    # live trading loop
    python main.py --dry-run          # simulate trades, no real orders
    python main.py --scan-only        # score markets, no trading
    python main.py --analyze-wallets  # run wallet analysis then exit
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Project imports ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from agents.scorer_agent import ScorerAgent, ScoredMarket
from agents.strategy_agent import StrategyAgent, TradeSignal, Position
from agents.executor_agent import ExecutorAgent
from agents.claude_analyst import ClaudeAnalyst
from agents.data_agent import DataAgent

# ── Logging setup ──────────────────────────────────────────────────────────
LOG_FILE = settings.logs_dir / "trading.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")
console = Console()


# ── Dashboard rendering ────────────────────────────────────────────────────

def _build_positions_table(positions: list[dict[str, Any]]) -> Table:
    t = Table(title="Open Positions", expand=True, border_style="blue")
    t.add_column("Market", style="cyan", max_width=40)
    t.add_column("Side", justify="center")
    t.add_column("Entry", justify="right")
    t.add_column("Current", justify="right")
    t.add_column("P&L %", justify="right")
    t.add_column("Hold (h)", justify="right")

    for p in positions:
        entry = float(p.get("entry_price", 0))
        current = float(p.get("current_price", entry))
        side = p.get("side", "?")
        if side == "YES":
            pnl_pct = (current - entry) / entry * 100 if entry else 0
        else:
            pnl_pct = (entry - current) / entry * 100 if entry else 0
        hold_h = (time.time() - float(p.get("opened_at", time.time()))) / 3600
        color = "green" if pnl_pct >= 0 else "red"
        t.add_row(
            p.get("question", "")[:40],
            side,
            f"{entry:.4f}",
            f"{current:.4f}",
            Text(f"{pnl_pct:+.1f}%", style=color),
            f"{hold_h:.1f}",
        )
    return t


def _build_top_markets_table(scored: list[ScoredMarket]) -> Table:
    t = Table(title="Top Scored Markets", expand=True, border_style="green")
    t.add_column("Market", style="cyan", max_width=45)
    t.add_column("YES", justify="right")
    t.add_column("Score", justify="right")
    t.add_column("Liq $", justify="right")
    t.add_column("Vol 24h $", justify="right")
    t.add_column("Days", justify="right")

    for sm in scored[:10]:
        m = sm.market
        t.add_row(
            m.question[:45],
            f"{m.yes_price:.3f}",
            f"{sm.score:.1f}",
            f"{m.liquidity:,.0f}",
            f"{m.volume_24h:,.0f}",
            f"{m.time_to_resolution_days:.1f}" if m.resolution_ts > 0 else "?",
        )
    return t


def _build_stats_panel(
    balance: float,
    win_rate: float,
    open_count: int,
    last_scan: str,
    mode: str,
) -> Panel:
    lines = [
        f"[bold]Mode:[/bold]       {mode}",
        f"[bold]Balance:[/bold]    ${balance:,.2f} USDC",
        f"[bold]Win rate:[/bold]   {win_rate*100:.1f}%",
        f"[bold]Open pos:[/bold]   {open_count}",
        f"[bold]Last scan:[/bold]  {last_scan}",
    ]
    return Panel("\n".join(lines), title="System Stats", border_style="yellow")


def _render_dashboard(
    positions: list[dict[str, Any]],
    top_markets: list[ScoredMarket],
    balance: float,
    win_rate: float,
    last_scan: str,
    mode: str,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="stats", size=8),
        Layout(name="positions", ratio=1),
        Layout(name="markets", ratio=1),
    )
    layout["stats"].update(
        _build_stats_panel(balance, win_rate, len(positions), last_scan, mode)
    )
    if positions:
        layout["positions"].update(_build_positions_table(positions))
    else:
        layout["positions"].update(Panel("No open positions", border_style="blue"))
    if top_markets:
        layout["markets"].update(_build_top_markets_table(top_markets))
    else:
        layout["markets"].update(Panel("No markets scored yet", border_style="green"))
    return layout


# ── Trading loop ───────────────────────────────────────────────────────────

class TradingOrchestrator:
    def __init__(
        self,
        dry_run: bool = False,
        scan_only: bool = False,
    ) -> None:
        self.dry_run = dry_run
        self.scan_only = scan_only
        self._running = True
        self._last_scan_time = "never"
        self._top_markets: list[ScoredMarket] = []
        self._wallet_patterns: dict[str, Any] = {}

        # Initialise agents
        self.scorer = ScorerAgent(
            min_liquidity=settings.min_liquidity,
            clob_url=settings.clob_url,
            gamma_url=settings.gamma_url,
        )
        self.strategy = StrategyAgent(
            exit_threshold=settings.exit_threshold,
            loss_cut=settings.loss_cut,
            volume_spike_multiplier=settings.volume_spike_multiplier,
            max_position_usdc=settings.max_position_size_usdc,
            db_path=str(settings.db_path),
            clob_url=settings.clob_url,
        )
        self.executor = ExecutorAgent(
            api_key=settings.poly_api_key,
            api_secret=settings.poly_api_secret,
            api_passphrase=settings.poly_api_passphrase,
            private_key=settings.poly_private_key,
            db_path=str(settings.db_path),
            max_position_usdc=settings.max_position_size_usdc,
            dry_run=dry_run,
        )
        self.analyst = ClaudeAnalyst(
            anthropic_api_key=settings.anthropic_api_key,
            openai_api_keys=settings.openai_api_keys,
        )
        self.data_agent = DataAgent(data_dir=settings.data_dir)

        logger.info(
            "Orchestrator initialised | dry_run=%s scan_only=%s", dry_run, scan_only
        )

    def stop(self) -> None:
        logger.info("Shutdown signal received")
        self._running = False

    # ── Single scan cycle ──────────────────────────────────────────────────

    async def _run_scan_cycle(self) -> None:
        """One full scan: score markets, generate signals, manage positions."""
        logger.info("=== Scan cycle start ===")
        scan_start = time.time()

        # 1. Score markets
        try:
            top = await self.scorer.get_top_markets(n=20, min_score=70.0)
            self._top_markets = top
            logger.info("Scored: %d qualifying markets", len(top))
        except Exception as exc:
            logger.error("Scorer failed: %s", exc)
            top = []

        if self.scan_only:
            self._last_scan_time = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            return

        # 2. Check exit conditions for open positions
        open_positions = self.strategy.get_open_positions()
        for pos in open_positions:
            try:
                should_exit, reason = await self.strategy.should_exit(pos)
                if should_exit:
                    self.executor.close_position(
                        position_id=pos.position_id,
                        exit_price=pos.current_price,
                        exit_reason=reason,
                    )
                    self.strategy.close_position(
                        market_id=pos.market_id,
                        exit_price=pos.current_price,
                        reason=reason,
                    )
            except Exception as exc:
                logger.error("Exit check failed for %s: %s", pos.market_id, exc)

        # 3. Generate new signals (skip markets already in position)
        open_market_ids = {p.market_id for p in self.strategy.get_open_positions()}
        signals: list[TradeSignal] = []

        for sm in top:
            if sm.market.condition_id in open_market_ids:
                continue
            # Claude analysis (cached 5 min)
            try:
                analysis = self.analyst.analyze_market(
                    {
                        "question": sm.market.question,
                        "yes_price": sm.market.yes_price,
                        "no_price": sm.market.no_price,
                        "volume_24h": sm.market.volume_24h,
                        "liquidity": sm.market.liquidity,
                        "score": sm.score,
                        "days_to_resolve": sm.market.time_to_resolution_days,
                    }
                )
            except Exception:
                analysis = {}

            # Skip if Claude recommends pass
            if analysis.get("recommendation") == "pass":
                continue

            signal = self.strategy.generate_signal(
                market=sm.market,
                patterns=self._wallet_patterns,
                score=sm.score,
                claude_analysis=analysis,
            )
            if signal:
                signals.append(signal)

        logger.info("Generated %d trade signals", len(signals))

        # 4. Execute top signals (cap at 3 new positions per cycle)
        for signal in signals[:3]:
            try:
                result = self.executor.place_order(signal)
                if result.status in ("filled", "MATCHED", "LIVE"):
                    fill_price = result.filled_price or signal.entry_price
                    self.strategy.open_position(signal, fill_price, result.size)
                    logger.info(
                        "New position opened: %s %s @ %.4f",
                        signal.side, signal.question[:50], fill_price,
                    )
            except Exception as exc:
                logger.error("Order execution failed: %s", exc)

        elapsed = time.time() - scan_start
        self._last_scan_time = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        logger.info("=== Scan cycle done in %.1fs ===", elapsed)

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        interval_s = settings.scan_interval_minutes * 60
        mode_label = (
            "DRY RUN" if self.dry_run else
            "SCAN ONLY" if self.scan_only else
            "LIVE"
        )
        console.print(
            f"\n[bold green]Polymarket AI Trading System[/bold green]  "
            f"mode=[bold]{mode_label}[/bold]  "
            f"scan_interval={settings.scan_interval_minutes}m\n"
        )

        # Load wallet patterns once at startup (from cache if available)
        try:
            self._wallet_patterns = self.data_agent.get_wallet_patterns()
            if not self._wallet_patterns:
                logger.info("No cached wallet patterns — using defaults")
                self._wallet_patterns = {
                    "median_captured_pct": 0.86,
                    "median_early_exit_pct": 0.91,
                    "median_loss_cut": 0.12,
                    "median_hold_hours": 4.0,
                }
        except Exception as exc:
            logger.warning("Wallet pattern load failed: %s", exc)
            self._wallet_patterns = {
                "median_captured_pct": 0.86,
                "median_early_exit_pct": 0.91,
            }

        with Live(console=console, refresh_per_second=0.5, screen=False) as live:
            next_scan = 0.0
            while self._running:
                now = time.time()
                if now >= next_scan:
                    try:
                        await self._run_scan_cycle()
                    except Exception as exc:
                        logger.error("Scan cycle error: %s", exc, exc_info=True)
                    next_scan = time.time() + interval_s

                # Refresh dashboard
                try:
                    positions_raw = self.executor.get_open_positions()
                    balance = self.executor.get_balance()
                    win_rate = self.executor.compute_win_rate()
                    layout = _render_dashboard(
                        positions=positions_raw,
                        top_markets=self._top_markets,
                        balance=balance,
                        win_rate=win_rate,
                        last_scan=self._last_scan_time,
                        mode=mode_label,
                    )
                    live.update(layout)
                except Exception as exc:
                    logger.debug("Dashboard render error: %s", exc)

                await asyncio.sleep(5)

        await self.scorer.close()
        console.print("\n[bold yellow]Trading system stopped.[/bold yellow]")


# ── Wallet analysis mode ───────────────────────────────────────────────────

async def run_wallet_analysis() -> None:
    console.print("[bold]Running full wallet analysis from poly_data …[/bold]")
    agent = DataAgent(data_dir=settings.data_dir)
    top_wallets = agent.analyze_top_wallets(force_refresh=True)
    console.print(f"Found [green]{len(top_wallets)}[/green] top wallets")

    patterns = agent.get_wallet_patterns()
    console.print("\n[bold]Behavioural patterns:[/bold]")
    for k, v in patterns.items():
        console.print(f"  {k}: {v}")

    signals = agent.find_alpha_signals()
    console.print("\n[bold]Alpha signals:[/bold]")
    for s in signals:
        console.print(f"  [{s['signal_type']}] {s['description']}")

    # Save to JSON
    import json
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_wallet_count": len(top_wallets),
        "patterns": patterns,
        "signals": signals,
        "top_wallets": top_wallets[:50],
    }
    settings.wallet_analysis_path.write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    console.print(f"\nSaved to [green]{settings.wallet_analysis_path}[/green]")


# ── Entry point ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polymarket AI Trading System")
    p.add_argument("--dry-run", action="store_true", help="Simulate trades without real orders")
    p.add_argument("--scan-only", action="store_true", help="Score markets but place no trades")
    p.add_argument("--analyze-wallets", action="store_true", help="Run wallet analysis then exit")
    return p.parse_args()


async def _main() -> None:
    args = parse_args()

    if args.analyze_wallets:
        await run_wallet_analysis()
        return

    orchestrator = TradingOrchestrator(
        dry_run=args.dry_run,
        scan_only=args.scan_only,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, orchestrator.stop)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(_main())
