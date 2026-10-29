"""V9 wrappers: a trend-regime gate and a buy-and-hold-relative training reward.

Neither wrapper touches prices, the ledger, fills or costs; the shared V4
environment stays the sole owner of those. The relative reward is a training
signal only: evaluation always uses the true NAV reward.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pandas as pd

FULL_EXPOSURE_ACTION = 2
REGIME_COLUMN = "trend_above_sma_200"


def _base(env):
    return env.unwrapped


class TrendGateWrapper(gym.Wrapper):
    """Force the full-exposure target while the observed trend flag is up.

    The flag (total-return feature price above its 200-session SMA at close t)
    lives in ``market_obs``, i.e. the observed stream an attacker may perturb;
    an attack harness that changes observed prices must recompute it. Below the
    trend line the agent's action passes through unchanged. The wrapped
    trade-decision rule still applies to the forced order.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if REGIME_COLUMN not in _base(env).market_obs.columns:
            raise KeyError(f"market_obs needs {REGIME_COLUMN} for the V9 gate")

    def _bull(self) -> bool:
        base = _base(self.env)
        return bool(base.market_obs.loc[base._decision_index, REGIME_COLUMN])

    def step(self, action):
        agent_action = int(action)
        forced = self._bull()
        chosen = FULL_EXPOSURE_ACTION if forced else agent_action
        obs, reward, terminated, truncated, info = self.env.step(chosen)
        info = dict(info)
        info.update({"agent_action": agent_action, "gate_forced_full": forced,
                     "gated_action": chosen})
        return obs, reward, terminated, truncated, info


class RelativeRewardWrapper(gym.Wrapper):
    """Training reward = log NAV change - log SPY total-return change.

    Both terms span the same close t -> close t+1 step and are known at close
    t+1, the moment the reward is delivered, so nothing from the future enters.
    The SPY term does not depend on the action, so the return-maximising policy
    is unchanged at the level of an exact finite-horizon objective, but
    finite-budget learning may differ. Its cumulative SPY reference is not
    the traded buy-and-hold ledger, which has costs and dividend timing.
    """

    def __init__(self, env: gym.Env, spy_log_return: pd.Series):
        super().__init__(env)
        index = pd.to_datetime(spy_log_return.index).normalize()
        self._spy = dict(zip(index, spy_log_return.to_numpy(dtype=float)))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        day = pd.Timestamp(info["valuation_date"]).normalize()
        market = self._spy.get(day)
        if market is None or not math.isfinite(market):
            raise KeyError(f"no SPY total return for {day.date()}")
        info = dict(info)
        info.update({"nav_log_reward": float(reward), "spy_log_return": market})
        return obs, float(reward) - market, terminated, truncated, info


def spy_log_returns(market_true: pd.DataFrame, cash_distribution: pd.Series) -> pd.Series:
    """log((close_t + ex-date cash_t) / close_{t-1}), indexed by date t."""
    close = market_true["close_raw"].astype(float).to_numpy()
    cash = np.asarray(cash_distribution, dtype=float)
    if len(cash) != len(close):
        raise ValueError("cash distributions must align with market_true")
    values = np.full(len(close), np.nan)
    values[1:] = np.log((close[1:] + cash[1:]) / close[:-1])
    return pd.Series(values, index=pd.to_datetime(market_true["date"]))
