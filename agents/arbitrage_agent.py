"""
ArbitrageAgent — catches price gaps between related markets.

Strategy: find correlated markets (same topic, different framing) where
YES prices don't sum to ~1.0.  For example:
  "Will X happen in Q1?" at 0.40
  "Will X happen in Q2?" at 0.45
  Sum = 0.85 — if these are truly mutually exclusive and exhaustive,
  one of them is mispriced.  We enter the cheaper one (higher expected value).

Correlation is detected by simple keyword overlap in market questions.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Maximum distance from 1.0 before we consider the pair mispriced
ARBITRAGE_THRESHOLD = 0.08  # sum deviates by >8% from 1.0
# Minimum keyword overlap ratio to call two markets "correlated"
MIN_KEYWORD_OVERLAP = 0.40


def _tokenize(text: str) -> set[str]:
    """Lower-case word tokens, stripping punctuation, dropping stop words."""
    stop = {
        "will", "the", "a", "an", "in", "of", "to", "and", "or",
        "be", "by", "is", "it", "at", "on", "for", "this", "that",
        "with", "from", "has", "have", "their", "its", "are", "was",
    }
    tokens = re.findall(r"[a-z]+", text.lower())
    return {t for t in tokens if t not in stop and len(t) > 2}


def _keyword_overlap(q1: str, q2: str) -> float:
    """Jaccard similarity between two question strings."""
    t1, t2 = _tokenize(q1), _tokenize(q2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


class ArbitrageAgent:
    """
    Identifies correlated market pairs where combined probabilities deviate
    significantly from 1.0, signalling a potential arbitrage entry.

    Usage:
        agent = ArbitrageAgent()
        pairs = agent.find_correlated_markets(markets_list)
        result = agent.evaluate(market, correlated_markets=pairs)
    """

    def __init__(
        self,
        arbitrage_threshold: float = ARBITRAGE_THRESHOLD,
        min_keyword_overlap: float = MIN_KEYWORD_OVERLAP,
    ) -> None:
        self.arbitrage_threshold = arbitrage_threshold
        self.min_keyword_overlap = min_keyword_overlap

    # ── Correlation detection ──────────────────────────────────────────────

    def find_correlated_markets(
        self, markets_list: list[Any]
    ) -> list[dict[str, Any]]:
        """
        Given a list of market objects, return pairs of correlated markets
        where the sum of YES prices deviates from 1.0 by more than the threshold.

        Each returned item:
            {
                "market_a": market_obj,
                "market_b": market_obj,
                "yes_sum": float,
                "deviation": float,   # |yes_sum - 1.0|
                "cheap_side": "a"|"b",
            }
        """
        pairs: list[dict[str, Any]] = []

        for i, m_a in enumerate(markets_list):
            q_a = getattr(m_a, "question", str(m_a))
            p_a = float(getattr(m_a, "yes_price", 0.5))

            for m_b in markets_list[i + 1 :]:
                q_b = getattr(m_b, "question", str(m_b))
                p_b = float(getattr(m_b, "yes_price", 0.5))

                # Check keyword similarity
                overlap = _keyword_overlap(q_a, q_b)
                if overlap < self.min_keyword_overlap:
                    continue

                yes_sum = p_a + p_b
                deviation = abs(yes_sum - 1.0)

                if deviation < self.arbitrage_threshold:
                    continue

                # The cheaper one (lower price relative to implied fair value)
                # In an overpriced pair, the one with lower price is cheap
                if yes_sum > 1.0:
                    # Both overpriced; cheapest is higher-priced one (closer to 1)
                    cheap_side = "b" if p_b > p_a else "a"
                else:
                    # Both underpriced vs combined; buy the cheaper one
                    cheap_side = "a" if p_a < p_b else "b"

                pairs.append(
                    {
                        "market_a": m_a,
                        "market_b": m_b,
                        "yes_sum": round(yes_sum, 4),
                        "deviation": round(deviation, 4),
                        "cheap_side": cheap_side,
                        "keyword_overlap": round(overlap, 3),
                    }
                )
                logger.debug(
                    "Arb pair: %s | %s  sum=%.3f dev=%.3f",
                    q_a[:40], q_b[:40], yes_sum, deviation,
                )

        logger.info("ArbitrageAgent found %d correlated pairs", len(pairs))
        return pairs

    # ── Signal evaluation ──────────────────────────────────────────────────

    def evaluate(
        self,
        market: Any,
        correlated_markets: list[dict[str, Any]] | None = None,
        all_markets: list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a single market for arbitrage opportunity.

        Args:
            market:             The market to evaluate.
            correlated_markets: Pre-computed pairs from find_correlated_markets().
                                If None and all_markets is provided, compute on the fly.
            all_markets:        Full market list (used if correlated_markets is None).

        Returns:
            {"action": "BUY"|"SELL"|"PASS", "confidence": float, "reason": str}
        """
        market_id: str = getattr(market, "condition_id", str(market))

        # Build pairs if not supplied
        if correlated_markets is None:
            if all_markets:
                correlated_markets = self.find_correlated_markets(all_markets)
            else:
                return {"action": "PASS", "confidence": 0.0, "reason": "no_market_data"}

        # Find pairs involving this market
        relevant = [
            p for p in correlated_markets
            if (
                getattr(p["market_a"], "condition_id", None) == market_id
                or getattr(p["market_b"], "condition_id", None) == market_id
            )
        ]

        if not relevant:
            return {"action": "PASS", "confidence": 0.0, "reason": "no_correlated_pair"}

        # Pick the best (highest deviation) pair
        best = max(relevant, key=lambda p: p["deviation"])
        deviation = best["deviation"]
        yes_sum = best["yes_sum"]

        is_market_a = getattr(best["market_a"], "condition_id", None) == market_id
        my_side = "a" if is_market_a else "b"
        partner = best["market_b"] if is_market_a else best["market_a"]
        partner_price = float(getattr(partner, "yes_price", 0.5))
        my_price = float(getattr(market, "yes_price", 0.5))

        if best["cheap_side"] == my_side:
            # This market is the cheap one in the mispriced pair → BUY
            action = "BUY"
            confidence = min(0.90, 0.5 + deviation * 3.0)
            reason = (
                f"arb: correlated pair YES sum={yes_sum:.3f} "
                f"(dev={deviation:.3f}); this={my_price:.3f} "
                f"partner={partner_price:.3f} → buy cheap leg"
            )
        else:
            # This market is the expensive leg — avoid
            action = "PASS"
            confidence = 0.1
            reason = (
                f"arb: this is the expensive leg "
                f"(sum={yes_sum:.3f}, dev={deviation:.3f})"
            )

        logger.info(
            "ArbitrageAgent %s: %s (conf=%.2f) — %s",
            market_id[:20], action, confidence, reason,
        )
        return {"action": action, "confidence": confidence, "reason": reason}
