"""
Kelly criterion position sizing utility.

Formula: f* = (p * b - q) / b
  p = estimated win probability
  b = net payout ratio = (1 / market_price) - 1
  q = 1 - p

Quarter-Kelly is applied (max_fraction default 0.25) to reduce variance.
Sweet spot for position sizing: f* between 0.05 and 0.15.
"""
from __future__ import annotations


def kelly_size(
    p_win: float,
    market_price: float,
    bankroll: float,
    max_fraction: float = 0.25,
) -> float:
    """
    Return the dollar amount to wager using (fractional) Kelly criterion.

    Args:
        p_win:        Estimated probability the bet wins (0-1).
        market_price: Current market price of the token (0-1).
                      E.g. 0.40 means the market prices YES at 40 cents.
        bankroll:     Total available capital in USDC.
        max_fraction: Cap as a fraction of bankroll (default 0.25 = quarter Kelly).

    Returns:
        Dollar amount to wager.  Returns 0.0 for negative-EV opportunities.
    """
    # Guard against degenerate inputs
    if not (0.0 < market_price < 1.0):
        return 0.0
    if not (0.0 < p_win < 1.0):
        return 0.0
    if bankroll <= 0.0:
        return 0.0

    # Net payout: for every $1 wagered at market_price, you win b dollars net
    b = (1.0 / market_price) - 1.0
    q = 1.0 - p_win

    # Kelly fraction
    f_star = (p_win * b - q) / b

    # Negative EV — do not trade
    if f_star <= 0.0:
        return 0.0

    # Apply quarter-Kelly cap
    f_capped = min(f_star, max_fraction)

    return round(f_capped * bankroll, 2)
