"""Gymnasium environment for the frozen single-asset V1 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from victimagent.features import (
    AccountState,
    ObservationBuilderV1,
)
from victimagent.ledger import PortfolioLedger


ACTION_WEIGHTS = (0.0, 0.5, 1.0)


@dataclass(frozen=True)
class DividendEvent:
    event_id: str
    ex_date: pd.Timestamp | str
    pay_date: pd.Timestamp | str
    per_share_amount: float


@dataclass(frozen=True)
class SplitEvent:
    event_id: str
    date: pd.Timestamp | str
    ratio: float


class SingleAssetTradingEnvV1(gym.Env[np.ndarray, int]):
    """Causal close-decision / next-open-execution trading environment.

    A call to ``step(action_t)`` uses an observation frozen after close ``t``.
    The target is executed at true open ``t+1``.  Reward is the log change from
    pre-trade open ``t+1`` to pre-trade open ``t+2``.  The returned observation
    contains only information available after close ``t+1``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_true: pd.DataFrame,
        market_obs: pd.DataFrame,
        observation_builder: ObservationBuilderV1,
        *,
        initial_cash: float = 1_000_000.0,
        transaction_cost_bps: float = 10.0,
        episode_length: int | None = None,
        random_start: bool = False,
        dividend_events: Iterable[DividendEvent] = (),
        split_events: Iterable[SplitEvent] = (),
        action_weights: Iterable[float] | None = None,
    ) -> None:
        super().__init__()
        # The observation builder defines the feature layout (which market
        # features, whether the cash rate is present). The engine below is
        # identical for every layout, so read the columns from the builder
        # rather than the frozen module constants.
        self.observation_builder = observation_builder
        self.market_true = self._validate_true_frame(market_true)
        self.market_obs = self._validate_observed_frame(
            market_obs, observation_builder.context_features
        )
        if not self.market_true["date"].equals(self.market_obs["date"]):
            raise ValueError("market_true and market_obs dates must match exactly")
        if len(self.market_true) < 3:
            raise ValueError("at least three market rows are required")
        if initial_cash <= 0 or not np.isfinite(initial_cash):
            raise ValueError("initial_cash must be finite and positive")
        if transaction_cost_bps < 0 or not np.isfinite(transaction_cost_bps):
            raise ValueError("transaction_cost_bps must be finite and non-negative")
        if episode_length is not None:
            if isinstance(episode_length, bool) or not isinstance(episode_length, int):
                raise TypeError("episode_length must be an integer or None")
            if episode_length < 1 or episode_length > len(self.market_true) - 2:
                raise ValueError("episode_length must fit within available transitions")
        if random_start and episode_length is None:
            raise ValueError("random_start requires a fixed episode_length")

        # Action representation may vary across studies (e.g. binary {0,1} vs
        # ternary {0,.5,1}); the ledger, reward and execution engine below do
        # not. The default reproduces the frozen V1 action set exactly.
        if action_weights is None:
            self._action_weights = ACTION_WEIGHTS
        else:
            weights = tuple(float(w) for w in action_weights)
            if len(weights) < 2:
                raise ValueError("action_weights must offer at least two targets")
            if not all(np.isfinite(w) and 0.0 <= w <= 1.0 for w in weights):
                raise ValueError("action_weights must be finite and within [0, 1]")
            self._action_weights = weights

        self.initial_cash = float(initial_cash)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.episode_length = episode_length
        self.random_start = bool(random_start)
        self._dividends = self._index_dividends(dividend_events)
        self._splits = self._index_splits(split_events)

        self.action_space = spaces.Discrete(len(self._action_weights))
        float_limit = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-float_limit,
            high=float_limit,
            shape=(len(self.observation_builder.observation_features),),
            dtype=np.float32,
        )

        self.ledger: PortfolioLedger | None = None
        self._decision_index = 0
        self._prepared_open_index: int | None = None
        self._episode_end_decision_index = len(self.market_true) - 3
        self._terminated = False
        self._truncated = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            unsupported = set(options) - {"start_index", "episode_length"}
            if unsupported:
                raise ValueError(f"unsupported reset options: {sorted(unsupported)}")
            explicit_start = "start_index" in options
            start_index = int(options.get("start_index", 0))
            effective_length = options.get("episode_length", self.episode_length)
        else:
            explicit_start = False
            start_index = 0
            effective_length = self.episode_length
        if effective_length is not None:
            effective_length = int(effective_length)
            if effective_length < 1 or effective_length > len(self.market_true) - 2:
                raise ValueError("episode_length must fit within available transitions")
            max_start = len(self.market_true) - effective_length - 2
            if self.random_start and not explicit_start:
                start_index = int(self.np_random.integers(0, max_start + 1))
            if start_index > max_start:
                raise ValueError("start_index plus episode_length exceeds available data")
            episode_end = start_index + effective_length - 1
        else:
            episode_end = len(self.market_true) - 3
        if start_index < 0 or start_index > len(self.market_true) - 3:
            raise ValueError("start_index must leave two future opens for reward")

        self.ledger = PortfolioLedger(self.initial_cash)
        self._decision_index = start_index
        self._prepared_open_index = None
        self._episode_end_decision_index = episode_end
        self._terminated = False
        self._truncated = False
        initialization_events: list[dict[str, Any]] = []
        for index in range(start_index + 1):
            initialization_events.extend(self._prepare_open(index))
        observation = self._compose_observation(start_index)
        return observation, {
            "decision_date": self._date(start_index),
            "decision_index": start_index,
            "episode_length": episode_end - start_index + 1,
            "episode_end_decision_index": episode_end,
            "initialization_events": initialization_events,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.ledger is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after episode termination or truncation")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid discrete action: {action!r}")
        return self._step_target_weight(
            self._action_weights[int(action)], requested_action=int(action), hold_shares=False
        )

    def step_target_weight(
        self, target_weight: float | None
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Audit/baseline interface; ``None`` means hold the current shares."""

        if target_weight is not None:
            if not np.isfinite(target_weight) or not 0.0 <= target_weight <= 1.0:
                raise ValueError("target_weight must be within [0, 1] or None")
        return self._step_target_weight(
            target_weight,
            requested_action="HOLD_SHARES" if target_weight is None else float(target_weight),
            hold_shares=target_weight is None,
        )

    def _step_target_weight(
        self,
        target_weight: float | None,
        *,
        requested_action: int | float | str,
        hold_shares: bool,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.ledger is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after episode termination or truncation")

        decision_i = self._decision_index
        execution_i = decision_i + 1
        valuation_i = decision_i + 2

        start_events = self._prepare_open(execution_i)
        start_snapshot = self.ledger.snapshot(self._true_open(execution_i))
        dividends_before_interval = self.ledger.cumulative_dividends
        if hold_shares:
            trade = self.ledger.hold_position(self._true_open(execution_i))
        else:
            assert target_weight is not None
            trade = self.ledger.execute_target(
                float(target_weight),
                self._true_open(execution_i),
                transaction_cost_bps=self.transaction_cost_bps,
            )

        # Freeze the next decision observation at close t+1 before advancing to
        # open t+2.  This prevents t+2 prices/events from entering that state.
        next_observation = self._compose_observation(execution_i)

        calendar_days = (self._date(valuation_i) - self._date(execution_i)).days
        annual_yield_pct = float(self.market_true.loc[decision_i, "dgs3mo_pct"])
        cash_benchmark_return = (1.0 + annual_yield_pct / 100.0) ** (
            calendar_days / 365.0
        ) - 1.0
        interest = self.ledger.accrue_cash(
            annual_yield_pct=annual_yield_pct,
            calendar_days=calendar_days,
        )
        end_events = self._prepare_open(valuation_i)
        end_snapshot = self.ledger.snapshot(self._true_open(valuation_i))
        reward = PortfolioLedger.open_to_open_reward(
            start_snapshot.nav, end_snapshot.nav
        )

        dividend_income = self.ledger.cumulative_dividends - dividends_before_interval
        net_pnl = end_snapshot.nav - start_snapshot.nav
        gross_pnl = net_pnl + trade.transaction_cost
        turnover = abs(trade.trade_notional) / start_snapshot.nav
        net_return = net_pnl / start_snapshot.nav
        gross_return = gross_pnl / start_snapshot.nav
        market_pnl = gross_pnl - interest - dividend_income

        self._decision_index += 1
        reached_episode_boundary = (
            self._decision_index > self._episode_end_decision_index
        )
        reached_natural_data_end = valuation_i == len(self.market_true) - 1
        self._terminated = reached_episode_boundary and reached_natural_data_end
        self._truncated = reached_episode_boundary and not reached_natural_data_end
        info = {
            "decision_date": self._date(decision_i),
            "execution_date": self._date(execution_i),
            "valuation_date": self._date(valuation_i),
            "requested_action": requested_action,
            "target_weight": trade.target_weight,
            "realized_weight": trade.realized_weight,
            "position_before_shares": trade.shares_pre,
            "position_after_shares": trade.shares_post,
            "true_execution_open": self._true_open(execution_i),
            "true_valuation_open": self._true_open(valuation_i),
            "start_pretrade_nav": start_snapshot.nav,
            "start_posttrade_nav": trade.nav_open_post,
            "end_pretrade_nav": end_snapshot.nav,
            "portfolio_value": end_snapshot.nav,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "market_pnl": market_pnl,
            "gross_return": gross_return,
            "net_return": net_return,
            "reward": float(reward),
            "turnover": turnover,
            "transaction_cost": trade.transaction_cost,
            "cash_interest": interest,
            "cash_benchmark_return": cash_benchmark_return,
            "dividend_income": dividend_income,
            "calendar_days": calendar_days,
            "trade": asdict(trade),
            "start_open_events": start_events,
            "end_open_events": end_events,
            "episode_end_reason": (
                "data_end"
                if self._terminated
                else "time_limit"
                if self._truncated
                else None
            ),
        }
        return next_observation, float(reward), self._terminated, self._truncated, info

    def _compose_observation(self, index: int) -> np.ndarray:
        if self.ledger is None:
            raise RuntimeError("ledger is not initialized")
        row = self.market_obs.loc[index]
        account = AccountState(
            cash_available=self.ledger.cash_available,
            shares=self.ledger.shares,
            dividend_receivable=self.ledger.dividend_receivable,
        )
        return self.observation_builder.compose(
            row,
            account,
            observed_close=float(row["observed_close"]),
        )

    def _prepare_open(self, index: int) -> list[dict[str, Any]]:
        if self.ledger is None:
            raise RuntimeError("ledger is not initialized")
        if self._prepared_open_index == index:
            return []
        if self._prepared_open_index is not None and index < self._prepared_open_index:
            raise RuntimeError("open processing cannot move backward")

        date = self._date(index)
        audit: list[dict[str, Any]] = []
        for event in self._splits.get(date, ()):
            shares_before = self.ledger.shares
            shares_after = self.ledger.apply_split(event.ratio)
            audit.append(
                {
                    "kind": "split",
                    "event_id": event.event_id,
                    "ratio": float(event.ratio),
                    "shares_before": shares_before,
                    "shares_after": shares_after,
                }
            )
        for event in self._dividends.get(date, {}).get("ex", ()):
            entitlement_shares = self.ledger.shares
            amount = self.ledger.create_dividend_receivable(
                event.event_id,
                per_share_amount=event.per_share_amount,
                entitlement_shares=entitlement_shares,
            )
            audit.append(
                {
                    "kind": "dividend_ex",
                    "event_id": event.event_id,
                    "entitlement_shares": entitlement_shares,
                    "amount": amount,
                }
            )
        for event in self._dividends.get(date, {}).get("pay", ()):
            amount = self.ledger.settle_dividend(event.event_id)
            audit.append(
                {"kind": "dividend_payment", "event_id": event.event_id, "amount": amount}
            )
        self._prepared_open_index = index
        return audit

    def _index_dividends(
        self, events: Iterable[DividendEvent]
    ) -> dict[pd.Timestamp, dict[str, list[DividendEvent]]]:
        valid_dates = set(self.market_true["date"])
        result: dict[pd.Timestamp, dict[str, list[DividendEvent]]] = {}
        ids: set[str] = set()
        for raw in events:
            event = DividendEvent(
                event_id=raw.event_id,
                ex_date=pd.Timestamp(raw.ex_date).normalize(),
                pay_date=pd.Timestamp(raw.pay_date).normalize(),
                per_share_amount=float(raw.per_share_amount),
            )
            if not event.event_id or event.event_id in ids:
                raise ValueError("dividend event IDs must be unique and non-empty")
            if event.ex_date not in valid_dates:
                raise ValueError("dividend ex_date must exist in market_true")
            if event.pay_date < event.ex_date:
                raise ValueError("dividend pay_date cannot precede ex_date")
            if event.per_share_amount < 0 or not np.isfinite(event.per_share_amount):
                raise ValueError("dividend amount must be finite and non-negative")
            ids.add(event.event_id)
            result.setdefault(event.ex_date, {"ex": [], "pay": []})["ex"].append(event)
            # A receivable can remain outstanding beyond the episode boundary.
            # Only schedule settlement when its mapped payment session exists.
            if event.pay_date in valid_dates:
                result.setdefault(event.pay_date, {"ex": [], "pay": []})["pay"].append(event)
        return result

    def _index_splits(
        self, events: Iterable[SplitEvent]
    ) -> dict[pd.Timestamp, list[SplitEvent]]:
        valid_dates = set(self.market_true["date"])
        result: dict[pd.Timestamp, list[SplitEvent]] = {}
        ids: set[str] = set()
        for raw in events:
            event = SplitEvent(
                event_id=raw.event_id,
                date=pd.Timestamp(raw.date).normalize(),
                ratio=float(raw.ratio),
            )
            if not event.event_id or event.event_id in ids:
                raise ValueError("split event IDs must be unique and non-empty")
            if event.date not in valid_dates:
                raise ValueError("split date must exist in market_true")
            if event.ratio <= 0 or not np.isfinite(event.ratio):
                raise ValueError("split ratio must be finite and positive")
            ids.add(event.event_id)
            result.setdefault(event.date, []).append(event)
        return result

    @staticmethod
    def _validate_true_frame(frame: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "open_raw", "close_raw", "dgs3mo_pct"}
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"market_true missing columns: {sorted(missing)}")
        result = frame.copy().reset_index(drop=True)
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
        if result["date"].duplicated().any() or not result["date"].is_monotonic_increasing:
            raise ValueError("market_true dates must be unique and increasing")
        for column in ("open_raw", "close_raw"):
            values = pd.to_numeric(result[column], errors="coerce")
            if values.isna().any() or (~np.isfinite(values)).any() or (values <= 0).any():
                raise ValueError(f"{column} must be finite and positive")
            result[column] = values.astype(float)
        rates = pd.to_numeric(result["dgs3mo_pct"], errors="coerce")
        if rates.isna().any() or (~np.isfinite(rates)).any() or (rates <= -100).any():
            raise ValueError("dgs3mo_pct must be finite and greater than -100")
        result["dgs3mo_pct"] = rates.astype(float)
        return result

    @staticmethod
    def _validate_observed_frame(
        frame: pd.DataFrame, context_features: Iterable[str]
    ) -> pd.DataFrame:
        context = list(context_features)
        required = {"date", "observed_close", *context}
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"market_obs missing columns: {sorted(missing)}")
        result = frame.copy().reset_index(drop=True)
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
        if result["date"].duplicated().any() or not result["date"].is_monotonic_increasing:
            raise ValueError("market_obs dates must be unique and increasing")
        numeric = result[["observed_close", *context]].apply(
            pd.to_numeric, errors="coerce"
        )
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("market_obs numeric inputs must be finite")
        if (numeric["observed_close"] <= 0).any():
            raise ValueError("observed_close must be positive")
        result.loc[:, ["observed_close", *context]] = numeric
        return result

    def _date(self, index: int) -> pd.Timestamp:
        return pd.Timestamp(self.market_true.loc[index, "date"])

    def _true_open(self, index: int) -> float:
        return float(self.market_true.loc[index, "open_raw"])
