from dataclasses import replace
import gymnasium as gym
import numpy as np
import pandas as pd

from victimagent.v4.environment import SingleAssetTradingEnvV4
from victimagent.v3.execution import TradeDecisionEnv
from victimagent.v9.wrappers import TrendGateWrapper
from victimagent.features import AccountState
from .policy import window_probabilities

class CachedTrainingV4(SingleAssetTradingEnvV4):
    """Read-only lookup cache; inherited V4 still owns all execution/accounting.

    Used only for clean training clones, never for mutable attack/scoring data.
    Reset still processes the same historical events with zero starting shares.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._dates_cache=tuple(pd.to_datetime(self.market_true.date))
        self._opens_cache=self.market_true.open_raw.to_numpy(dtype=float)
        self._closes_cache=self.market_true.close_raw.to_numpy(dtype=float)
        self._observations_cache=self.market_obs.to_dict('records')

    def _date(self,index):
        return self._dates_cache[index] if hasattr(self,'_dates_cache') else super()._date(index)

    def _true_open(self,index):
        return float(self._opens_cache[index]) if hasattr(self,'_opens_cache') else super()._true_open(index)

    def _true_close(self,index):
        return float(self._closes_cache[index]) if hasattr(self,'_closes_cache') else super()._true_close(index)

    def _compose_observation(self,index):
        if not hasattr(self,'_observations_cache'):
            return super()._compose_observation(index)
        row=self._observations_cache[index]
        account=AccountState(cash_available=self.ledger.cash_available,shares=self.ledger.shares,
                             dividend_receivable=self.ledger.dividend_receivable)
        return self.observation_builder.compose(row,account,observed_close=float(row['observed_close']))

def trading_environment(bundle, *, training=False, cached=False):
    cls=CachedTrainingV4 if cached else SingleAssetTradingEnvV4
    base = cls(
        bundle.market_true, bundle.market_obs, bundle.observation_builder,
        episode_length=63 if training else None, random_start=False,
        initial_cash=1_000_000., transaction_cost_bps=10.,
        dividend_events=bundle.dividend_events, split_events=bundle.split_events,
        action_weights=(0., .5, 1.))
    return TrendGateWrapper(TradeDecisionEnv(base, 'v3_min_hold_five'))

class MixedReward(gym.Wrapper):
    def __init__(self, env, spy, mixed):
        super().__init__(env)
        self.spy = dict(zip(pd.to_datetime(spy.index).normalize(), spy.to_numpy()))
        self.mixed = mixed
        self.streak = 0

    def reset(self, **kwargs):
        self.streak = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, nav, terminated, truncated, info = self.env.step(action)
        spy = float(self.spy[pd.Timestamp(info['valuation_date']).normalize()])
        if not np.isfinite(spy) or not np.isfinite(nav):
            raise ValueError('nonfinite reward')
        self.streak = self.streak+1 if nav > 0 else 0
        bonus = .1*max(min(self.streak,5)-1,0)*max(nav,0)
        weighted = .5*bonus if self.mixed else 0.
        reward = nav-.2*spy+weighted if self.mixed else nav
        info = dict(info, nav_log_reward=float(nav), spy_log_return=spy,
                    relative_log_reward=nav-spy, positive_return_streak=self.streak,
                    positive_streak_bonus=bonus, weighted_streak_bonus=weighted,
                    trend_component_reward=nav+bonus, training_reward=reward,
                    reward_mode='mix' if self.mixed else 'nav')
        return obs, reward, terminated, truncated, info

class RecentWindowEnv(gym.Wrapper):
    def __init__(self, bundle, spy, mixed, seed):
        super().__init__(MixedReward(trading_environment(bundle, training=True,cached=True), spy, mixed))
        self.starts, self.ages, self.probabilities = window_probabilities(len(bundle.market_true))
        self.sampler = np.random.default_rng(seed)
        self.sampled_ages = []
        self.transition_dates = []
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.sampler = np.random.default_rng(seed)
        selected = int(self.sampler.choice(self.starts, p=self.probabilities))
        self.sampled_ages.append(int(self.ages[selected]))
        self.steps = 0
        return self.env.reset(seed=seed, options={'start_index':selected, 'episode_length':63})

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        self.transition_dates.append(str(info['valuation_date'].date()))
        if self.steps == 63:
            # Even the newest window's endpoint is an artificial training
            # horizon: always bootstrap it, including when it equals cutoff.
            terminated, truncated = False, True
            info = dict(info, episode_end_reason='time_limit')
        return obs, reward, terminated, truncated, info
