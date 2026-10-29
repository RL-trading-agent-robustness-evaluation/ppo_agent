"""V12 benchmark only: reuse V11 event ordering; never load or execute PPO."""
from dataclasses import asdict
import math
import numpy as np
import pandas as pd
from victimagent.ledger import PortfolioLedger, TradeExecution
from victimagent.v4.environment import SingleAssetTradingEnvV4
from victimagent.v4.evaluation import metrics


class FractionalBudgetLedger(PortfolioLedger):
    """Isolated fractional-share variant; inherited cash/event semantics unchanged."""
    budget = 0.0

    @staticmethod
    def _validate_integer_shares(shares):
        if isinstance(shares, bool) or not isinstance(shares, (int, float)):
            raise TypeError('shares must be numeric')
        if not math.isfinite(shares) or shares < 0:
            raise ValueError('shares must be finite and nonnegative')

    def apply_split(self, ratio):
        self._validate_positive_price(ratio)
        self.shares *= ratio
        self._assert_invariants()
        return self.shares

    def execute_target(self, target_weight, true_open, *, transaction_cost_bps=10.0):
        # The inherited environment invokes this hook only on scheduled buy days.
        self._validate_positive_price(true_open)
        self._validate_finite('budget', self.budget)
        self._validate_finite('cost', transaction_cost_bps)
        if self.budget <= 0 or transaction_cost_bps < 0:
            raise ValueError('invalid purchase budget/cost')
        if self.cash_available + self.tolerance < self.budget:
            raise RuntimeError('fixed budget cannot be funded; never silently reduce it')
        pre = self.snapshot(true_open)
        notional = self.budget / (1 + transaction_cost_bps / 10000)
        cost = self.budget - notional
        quantity = notional / true_open
        old_shares = self.shares
        old_cash = self.cash_available
        self.cash_available -= self.budget
        self.shares += quantity
        self.cumulative_transaction_cost += cost
        self._assert_invariants()
        post = self.snapshot(true_open)
        if not math.isclose(post.nav, pre.nav-cost, rel_tol=1e-12, abs_tol=1e-8):
            raise AssertionError('purchase NAV conservation')
        return TradeExecution(post.market_value/post.nav, post.market_value/post.nav,
            old_shares, self.shares, self.shares, quantity, true_open, notional,
            cost, old_cash, self.cash_available, pre.nav, post.nav)


class MonthlyEnvironment(SingleAssetTradingEnvV4):
    def _compose_observation(self, index):
        # Benchmark orders use only the frozen calendar/budget, never policy
        # observations. Parent account-feature validation requires integers;
        # do not alter it or coerce fractional holdings to fake integer shares.
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        replacement = FractionalBudgetLedger(self.initial_cash)
        replacement.__dict__.update(self.ledger.__dict__)
        replacement.shares = float(replacement.shares)
        self.ledger = replacement
        return observation, info


def monthly_schedule(dates, initial_cash=1_000_000.0):
    """Dates[0] is the starting close: no order can execute at its earlier open."""
    dates = pd.DatetimeIndex(dates)
    if not dates.is_monotonic_increasing or not dates.is_unique or len(dates) < 2:
        raise ValueError('unique ascending sessions required')
    months = pd.period_range(dates[0].to_period('M'), dates[-1].to_period('M'), freq='M')
    result = {}
    allocated = 0.0
    for index, month in enumerate(months):
        candidates = [i for i in range(1, len(dates)) if dates[i].to_period('M') == month]
        if not candidates:
            raise ValueError(f'no executable session in {month}')
        i = candidates[0]
        budget = initial_cash/len(months) if index < len(months)-1 else initial_cash-allocated
        allocated += budget
        result[i] = dict(month=str(month), decision_date=str(dates[i-1].date()),
            execution_date=str(dates[i].date()), budget=budget)
    return result


def run_monthly(bundle, initial_cash=1_000_000.0):
    schedule = monthly_schedule(bundle.market_true.date, initial_cash)
    env = MonthlyEnvironment(bundle.market_true, bundle.market_obs, bundle.observation_builder,
        initial_cash=initial_cash, transaction_cost_bps=10.,
        dividend_events=bundle.dividend_events, split_events=bundle.split_events)
    _, reset = env.reset(seed=0)
    rows, purchases = [], []
    while True:
        i = env._decision_index + 1
        order = schedule.get(i)
        env.ledger.budget = order['budget'] if order else 0.
        _, reward, terminated, truncated, info = env.step_target_weight(1. if order else None)
        row = {**info, 'decision_date': str(info['decision_date'].date()),
            'execution_date': str(info['execution_date'].date()),
            'valuation_date': str(info['valuation_date'].date()),
            'reward_available_date': str(info['reward_available_date'].date()),
            'next_observation_date': str(info['next_observation_date'].date()),
            'start_nav': info['start_close_nav'], 'end_nav': info['end_close_nav'],
            'action': 2 if order else -1, 'trade_notional': info['trade']['trade_notional'],
            'terminated': terminated, 'truncated': truncated,
            'purchase_budget': env.ledger.budget, 'cash_available': env.ledger.cash_available,
            'shares': env.ledger.shares, 'dividend_receivable': env.ledger.dividend_receivable}
        rows.append(row)
        if order:
            purchases.append({**order, **asdict(env.ledger.snapshot(info['true_execution_open'])),
                'trade': info['trade']})
        if terminated or truncated:
            break
    error = sum(r['reward'] for r in rows)-math.log(rows[-1]['end_nav']/reset['initial_close_nav'])
    if truncated or not terminated or abs(error) > 1e-10:
        raise AssertionError('full horizon/reward identity')
    measures = metrics(rows)
    measures.update(final_nav=rows[-1]['end_nav'], final_cash=env.ledger.cash_available,
        final_shares=env.ledger.shares, final_receivable=env.ledger.dividend_receivable,
        total_cash_interest=env.ledger.cumulative_cash_interest,
        total_dividends=env.ledger.cumulative_dividends,
        principal_budget_spent=sum(p['budget'] for p in purchases), reward_nav_error=error)
    return rows, purchases, measures
