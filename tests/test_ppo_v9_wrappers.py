import math

import numpy as np
import pandas as pd
import pytest

from victimagent.v9.wrappers import spy_log_returns
from victim_ppo.v9.runner import responsiveness


def test_spy_reference_uses_close_plus_ex_date_cash():
    market = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        "close_raw": [100.0, 99.0, 101.0],
    })
    cash = pd.Series([0.0, 2.0, 0.0])
    result = spy_log_returns(market, cash)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(math.log(101.0 / 100.0))
    assert result.iloc[2] == pytest.approx(math.log(101.0 / 99.0))


def test_responsiveness_only_perturbs_market_dimensions():
    class ThresholdPolicy:
        def predict(self, matrix, deterministic=True):
            values = np.asarray(matrix)
            return (values[:, 0] > 0).astype(int), None

    observations = []
    for value in (-2.0, -1.0, 1.0, 2.0):
        row = np.arange(20, dtype=np.float32)
        row[0] = value
        observations.append(row)
    result = responsiveness(ThresholdPolicy(), observations)
    assert result["account_and_rule_dimensions_preserved"] is True
    assert result["action_change_ratio"] > 0
    assert result["eligible"] is True
