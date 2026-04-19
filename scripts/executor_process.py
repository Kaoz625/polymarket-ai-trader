"""
executor_process.py — Standalone executor process.

Reads data/thesis.json, runs consensus voting across 3 specialist agents
(WhaleAgent, ConvergenceAgent, ArbitrageAgent), places orders using
Kelly sizing.

Usage:
    python scripts/executor_process.py [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from agents.kelly import kelly_size
from agents.whale_agent import WhaleAgent
from agents.convergence_agent import ConvergenceAgent
from agents.arbitrage_agent import ArbitrageAgent
from agents.executor_agent import ExecutorAgent
from agents.strategy_agent import StrategyAgent, TradeSignal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("executor_process")

MIN_VOLUME_24H = 50_000  # USDC — skip markets below this volume


class _MarketProxy:
    """Minimal object to pass thesis dict fields to agent evaluate() methods."""

    def __init__(self, thesis: dict[str, Any]) -> None:
        self.condition_id: str = thesis.get("condition_id", "")
        self.question: str = thesis.get("question", "")
        self.yes_price: float = float(thesis.get("yes_price", 0.5))
        self.no_price: float = float(thesis.get("no_price", 0.5))
        self.volume_24h: float = float(thesis.get("volume_24h", 0))
        self.liquidity: float = float(thesis.get("liquidity", 0))
        self.resolution_ts: float = float(thesis.get("resolution_ts", 0))


class _SignalProxy:
    """Minimal TradeSignal-like object for ExecutorAgent.place_order()."""

    def __init__(
        self,
        market_id: str,
        question: str,
        side: str,
        entry_price: float,
        target_price: float,
        stop_price: float,
    ) -> None:
        self.market_id = market_id
        self.question = question
        self.side = side
        self.entry_price = entry_price
        self.target_price = target_price
        self.stop_price = stop_price


async def consensus_execute(
    thesis: dict[str, Any],
    whale_agent: WhaleAgent,
    convergence_agent: ConvergenceAgent,
    arbitrage_agent: ArbitrageAgent,
    executor: ExecutorAgent,
    wallet_balance: float,
    dry_run: bool,
) -> bool:
    """
    Collect votes from all 3 specialist agents and execute with Kelly sizing.

    Voting rules:
        2+ agree BUY → full Kelly position
        1 agrees BUY → half Kelly position
        0 agree     → no trade

    Returns True if an order was placed.
    """
    market = _MarketProxy(thesis)

    # Volume filter
    if market.volume_24h < MIN_VOLUME_24H:
        logger.info(
            "SKIP %s: volume %.0f < %.0f",
            market.question[:50], market.volume_24h, MIN_VOLUME_24H,
        )
        return False

    # Collect agent votes concurrently
    fair_value = float(thesis.get("fair_value_yes", market.yes_price))

    whale_task = asyncio.create_task(whale_agent.evaluate(market))
    # ConvergenceAgent is sync — run in executor
    loop = asyncio.get_event_loop()
    convergence_result = await loop.run_in_executor(
        None,
        lambda: convergence_agent.evaluate(market, fair_value_yes=fair_value),
    )
    arb_result = {"action": "PASS", "confidence": 0.0, "reason": "no_all_markets"}
    # ArbitrageAgent needs full market list — evaluate with empty list (graceful PASS)
    arb_result = arbitrage_agent.evaluate(market, correlated_markets=[])

    whale_result = await whale_task

    votes = [whale_result, convergence_result, arb_result]
    buy_votes = sum(1 for v in votes if v.get("action") == "BUY")

    logger.info(
        "Consensus for %s: %d/3 BUY votes  whale=%s conv=%s arb=%s",
        market.question[:40],
        buy_votes,
        whale_result.get("action"),
        convergence_result.get("action"),
        arb_result.get("action"),
    )

    if buy_votes == 0:
        logger.info("No consensus — skipping %s", market.question[:50])
        return False

    # Determine AI recommendation → side
    recommendation = thesis.get("recommendation", "pass")
    if recommendation == "enter_yes":
        side = "YES"
        entry_price = market.yes_price
    elif recommendation == "enter_no":
        side = "NO"
        entry_price = market.no_price
    else:
        logger.info("AI pass on %s — skipping", market.question[:50])
        return False

    # Kelly sizing
    p_win = float(thesis.get("combined_confidence", thesis.get("ai_confidence", 0.6)))
    full_kelly_amount = kelly_size(
        p_win=p_win,
        market_price=entry_price,
        bankroll=wallet_balance,
        max_fraction=0.25,
    )

    if buy_votes >= 2:
        size_usdc = full_kelly_amount
        label = "full Kelly"
    else:
        size_usdc = full_kelly_amount * 0.5
        label = "half Kelly"

    # Cap at settings max
    size_usdc = min(size_usdc, settings.max_position_size_usdc)

    if size_usdc <= 0:
        logger.info("Kelly returned 0 for %s — skipping", market.question[:50])
        return False

    # Compute target/stop from strategy logic
    loss_cut = settings.loss_cut
    stop_price = entry_price * (1 - loss_cut)
    target_price = min(entry_price + (1.0 - entry_price) * 0.86, 0.98)

    signal = _SignalProxy(
        market_id=market.condition_id,
        question=market.question,
        side=side,
        entry_price=entry_price,
        target_price=target_price,
        stop_price=stop_price,
    )

    logger.info(
        "EXECUTE %s %s @ %.4f  size=%.2f USDC (%s, %d/3 votes)",
        side, market.question[:40], entry_price, size_usdc, label, buy_votes,
    )

    result = executor.place_order(signal, size_usdc=size_usdc)
    logger.info(
        "Order result: status=%s filled_price=%.4f order_id=%s",
        result.status, result.filled_price, result.order_id,
    )
    return result.status in ("filled", "MATCHED", "LIVE")


async def main(dry_run: bool) -> None:
    logger.info("Executor process starting (dry_run=%s) …", dry_run)

    thesis_path = settings.data_dir / "thesis.json"
    if not thesis_path.exists():
        logger.error("thesis.json not found — run brain.py first")
        sys.exit(1)

    theses: list[dict] = json.loads(thesis_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d theses", len(theses))

    whale_agent = WhaleAgent(wallet_analysis_path=settings.wallet_analysis_path)
    convergence_agent = ConvergenceAgent(db_path=settings.db_path)
    arbitrage_agent = ArbitrageAgent()
    executor = ExecutorAgent(
        api_key=settings.poly_api_key,
        api_secret=settings.poly_api_secret,
        api_passphrase=settings.poly_api_passphrase,
        private_key=settings.poly_private_key,
        db_path=str(settings.db_path),
        max_position_usdc=settings.max_position_size_usdc,
        dry_run=dry_run,
    )

    wallet_balance = executor.get_balance()
    logger.info("Wallet balance: %.2f USDC", wallet_balance)

    placed = 0
    for thesis in theses:
        try:
            ok = await consensus_execute(
                thesis=thesis,
                whale_agent=whale_agent,
                convergence_agent=convergence_agent,
                arbitrage_agent=arbitrage_agent,
                executor=executor,
                wallet_balance=wallet_balance,
                dry_run=dry_run,
            )
            if ok:
                placed += 1
        except Exception as exc:
            logger.error(
                "Error executing thesis %s: %s",
                thesis.get("condition_id", "?"), exc,
            )

    await whale_agent.close()
    logger.info("Executor process complete: %d/%d orders placed", placed, len(theses))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polymarket executor process")
    p.add_argument("--dry-run", action="store_true", help="Simulate orders, no real trades")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(dry_run=args.dry_run))
