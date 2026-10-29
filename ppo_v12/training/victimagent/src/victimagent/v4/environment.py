"""Close-t decision, next-open fill, next-close reward and observation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from victimagent.environment import SingleAssetTradingEnvV1
from victimagent.ledger import PortfolioLedger


class SingleAssetTradingEnvV4(SingleAssetTradingEnvV1):
    """A separately versioned timeline using the frozen V1 event/ledger code.

    A decision uses data available after close t. Cash held at that close earns
    the lagged rate until the next session; events and the trade occur at the
    true open t+1. Both the reward and next observation become available at
    close t+1. The final decision is on the penultimate supplied session.
    """

    def __init__(self, market_true, market_obs, observation_builder, *,
                 initial_cash=1_000_000.0, transaction_cost_bps=10.0,
                 episode_length=None, random_start=False, dividend_events=(),
                 split_events=(), action_weights=None):
        # V1's episode bounds reserve a second future open. Do not alter V1:
        # configure V4's own one-session horizon after the common validation.
        super().__init__(
            market_true, market_obs, observation_builder,
            initial_cash=initial_cash, transaction_cost_bps=transaction_cost_bps,
            episode_length=None, random_start=False,
            dividend_events=dividend_events, split_events=split_events,
            action_weights=action_weights,
        )
        if episode_length is not None:
            if isinstance(episode_length, bool) or not isinstance(episode_length, int):
                raise TypeError("episode_length must be an integer or None")
            if not 1 <= episode_length <= len(self.market_true) - 1:
                raise ValueError("V4 episode_length must fit within available closes")
        if random_start and episode_length is None:
            raise ValueError("random_start requires a fixed episode_length")
        self.episode_length = episode_length
        self.random_start = bool(random_start)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None):
        # Invoke Gymnasium reset directly: V1's reset bounds are intentionally
        # different and would drop V4's last valid close-to-close transition.
        from gymnasium import Env

        Env.reset(self, seed=seed)
        options = options or {}
        unsupported = set(options) - {"start_index", "episode_length"}
        if unsupported:
            raise ValueError(f"unsupported reset options: {sorted(unsupported)}")
        length = options.get("episode_length", self.episode_length)
        if length is not None:
            if isinstance(length, bool) or not isinstance(length, int):
                raise TypeError("episode_length must be an integer or None")
            if not 1 <= length <= len(self.market_true) - 1:
                raise ValueError("V4 episode_length must fit within available closes")
            max_start = len(self.market_true) - length - 1
            start = (int(self.np_random.integers(0, max_start + 1))
                     if self.random_start and "start_index" not in options else
                     options.get("start_index", 0))
        else:
            max_start = len(self.market_true) - 2
            start = options.get("start_index", 0)
            length = len(self.market_true) - 1 - start
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start <= max_start:
            raise ValueError("invalid V4 start_index for requested episode")

        self.ledger = PortfolioLedger(self.initial_cash)
        self._decision_index = start
        self._prepared_open_index = None
        self._episode_end_decision_index = start + length - 1
        self._terminated = self._truncated = False
        initialization_events = []
        for i in range(start + 1):
            initialization_events.extend(self._prepare_open(i))
        return self._compose_observation(start), {
            "decision_date": self._date(start),
            "decision_index": start,
            "episode_length": length,
            "episode_end_decision_index": self._episode_end_decision_index,
            "initialization_events": initialization_events,
            "initial_close_nav": self.ledger.snapshot(self._true_close(start)).nav,
        }

    def _true_close(self, index: int) -> float:
        return float(self.market_true.loc[index, "close_raw"])

    def _step_target_weight(self, target_weight, *, requested_action,
                            hold_shares):
        if self.ledger is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after episode termination or truncation")

        decision_i = self._decision_index
        next_i = decision_i + 1
        start = self.ledger.snapshot(self._true_close(decision_i))
        if start.nav <= 0:
            raise RuntimeError("V4 start NAV must be positive")

        # The entire date-to-date ACT/365 cash period is credited to cash
        # already available at close t. Payment and sale proceeds at next open
        # do not receive interest retroactively.
        calendar_days = (self._date(next_i) - self._date(decision_i)).days
        annual_yield_pct = float(self.market_true.loc[decision_i, "dgs3mo_pct"])
        cash_benchmark_return = (1 + annual_yield_pct / 100) ** (calendar_days / 365) - 1
        interest = self.ledger.accrue_cash(annual_yield_pct, calendar_days)
        before_dividends = self.ledger.cumulative_dividends
        events = self._prepare_open(next_i)
        pre_open = self.ledger.snapshot(self._true_open(next_i))
        if hold_shares:
            trade = self.ledger.hold_position(self._true_open(next_i))
        else:
            assert target_weight is not None
            trade = self.ledger.execute_target(
                float(target_weight), self._true_open(next_i),
                transaction_cost_bps=self.transaction_cost_bps,
            )
        end = self.ledger.snapshot(self._true_close(next_i))
        reward = PortfolioLedger.open_to_open_reward(start.nav, end.nav)
        next_observation = self._compose_observation(next_i)
        net_pnl = end.nav - start.nav
        gross_pnl = net_pnl + trade.transaction_cost
        dividend_income = self.ledger.cumulative_dividends - before_dividends

        self._decision_index = next_i
        boundary = decision_i >= self._episode_end_decision_index
        self._terminated = boundary and next_i == len(self.market_true) - 1
        self._truncated = boundary and not self._terminated
        info = {
            "decision_date": self._date(decision_i),
            "execution_date": self._date(next_i),
            "valuation_date": self._date(next_i),
            "reward_available_date": self._date(next_i),
            "next_observation_date": self._date(next_i),
            "requested_action": requested_action,
            "target_weight": trade.target_weight,
            "realized_weight": trade.realized_weight,
            "position_before_shares": trade.shares_pre,
            "position_after_shares": trade.shares_post,
            "true_execution_open": self._true_open(next_i),
            "true_valuation_close": self._true_close(next_i),
            "start_close_nav": start.nav,
            "pretrade_open_nav": pre_open.nav,
            "posttrade_open_nav": trade.nav_open_post,
            "end_close_nav": end.nav,
            "portfolio_value": end.nav,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "market_pnl": gross_pnl - interest - dividend_income,
            "gross_return": gross_pnl / start.nav,
            "net_return": net_pnl / start.nav,
            "reward": float(reward),
            "turnover": abs(trade.trade_notional) / start.nav,
            "transaction_cost": trade.transaction_cost,
            "cash_interest": interest,
            "cash_benchmark_return": cash_benchmark_return,
            "dividend_income": dividend_income,
            "calendar_days": calendar_days,
            "trade": asdict(trade),
            "next_open_events": events,
            "episode_end_reason": "data_end" if self._terminated else
                                  "time_limit" if self._truncated else None,
        }
        return next_observation, float(reward), self._terminated, self._truncated, info
