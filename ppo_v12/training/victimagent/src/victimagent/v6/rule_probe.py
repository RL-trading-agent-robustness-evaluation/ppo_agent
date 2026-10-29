"""Single frozen SPY-first defensive rule for exploratory pre-2022 probing."""

from __future__ import annotations

import math


def choose_target(*, fast_above_slow: int, scaled_volatility: float,
                  train_volatility_median: float) -> int:
    """Action 1 is half SPY only in a downtrend with above-train-median vol.

    The inputs are close-t observations and a statistic fixed on this fold's
    train dates. Otherwise action 2 is fully invested in SPY.
    """
    if fast_above_slow not in (0, 1):
        raise ValueError("MA trend flag must be 0 or 1")
    if not all(math.isfinite(float(x)) for x in (scaled_volatility, train_volatility_median)):
        raise ValueError("Non-finite volatility or train threshold")
    return 1 if fast_above_slow == 0 and scaled_volatility > train_volatility_median else 2
