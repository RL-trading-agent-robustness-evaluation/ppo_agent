"""Deterministic synthetic market for the pre-DQN learnability gate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from victimagent.environment import SingleAssetTradingEnvV1
from victimagent.features import AccountState, CONTEXT_FEATURES, OBSERVATION_FEATURES


class ToyObservationBuilder:
    """Pass through seven toy context values and append live account weights."""

    context_features = CONTEXT_FEATURES
    observation_features = OBSERVATION_FEATURES

    def compose(self, row, account: AccountState, *, observed_close: float) -> np.ndarray:
        account.validate()
        nav = account.cash_available + account.shares * observed_close + account.dividend_receivable
        account_values = (
            account.shares * observed_close / nav,
            account.cash_available / nav,
            account.dividend_receivable / nav,
        )
        result = np.asarray(
            [*(float(row[name]) for name in CONTEXT_FEATURES), *account_values],
            dtype=np.float32,
        )
        if result.shape != (len(OBSERVATION_FEATURES),) or not np.isfinite(result).all():
            raise AssertionError("invalid toy observation")
        return result


def make_toy_market_env(
    *,
    rows: int = 514,
    episode_length: int = 64,
    random_start: bool = True,
) -> SingleAssetTradingEnvV1:
    """Return a production ledger/env with an obvious causal long/cash signal.

    Feature 0 at decision i is +1 or -1 and determines the true open-to-open
    move controlled by that action.  +1 makes 100% risky optimal; -1 makes cash
    optimal.  The market is deterministic, balanced, and has zero costs/rates.
    """

    if rows < episode_length + 2:
        raise ValueError("toy market needs episode_length + two future rows")
    signal = np.where(np.arange(rows) % 2 == 0, 1.0, -1.0)
    opens = np.full(rows, 100.0, dtype=float)
    for decision_i in range(rows - 2):
        opens[decision_i + 2] = opens[decision_i + 1] * (
            1.02 if signal[decision_i] > 0 else 0.98
        )
    dates = pd.bdate_range("2000-01-03", periods=rows)
    true = pd.DataFrame(
        {"date": dates, "open_raw": opens, "close_raw": opens, "dgs3mo_pct": 0.0}
    )
    observed = pd.DataFrame({"date": dates, "observed_close": opens})
    for index, name in enumerate(CONTEXT_FEATURES):
        observed[name] = signal if index == 0 else 0.0
    return SingleAssetTradingEnvV1(
        true,
        observed,
        ToyObservationBuilder(),
        initial_cash=1_000_000.0,
        transaction_cost_bps=0.0,
        episode_length=episode_length,
        random_start=random_start,
    )


def toy_optimal_action(observation: np.ndarray) -> int:
    return 2 if float(observation[0]) > 0 else 0
