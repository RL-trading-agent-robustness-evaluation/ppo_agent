"""Deterministic single-asset portfolio ledger.

The ledger is intentionally independent of observations and policies.  It only
accepts true execution/valuation prices, so an adversarial observation cannot
silently change fills, cash, dividends, or P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LedgerSnapshot:
    """Auditable portfolio state at one true market price."""

    true_price: float
    cash_available: float
    shares: int
    dividend_receivable: float
    market_value: float
    nav: float


@dataclass(frozen=True)
class TradeExecution:
    """Full audit record for one target-weight execution."""

    target_weight: float
    realized_weight: float
    shares_pre: int
    shares_target: int
    shares_post: int
    delta_shares: int
    true_open: float
    trade_notional: float
    transaction_cost: float
    cash_pre_trade: float
    cash_post_trade: float
    nav_open_pre: float
    nav_open_post: float


class PortfolioLedger:
    """Long-only, unlevered, integer-share ledger for one risky asset."""

    def __init__(
        self,
        initial_cash: float,
        initial_shares: int = 0,
        *,
        tolerance: float = 1e-9,
    ) -> None:
        self._validate_finite("initial_cash", initial_cash)
        self._validate_integer_shares(initial_shares)
        if initial_cash < 0:
            raise ValueError("initial_cash must be non-negative")
        if tolerance <= 0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be finite and positive")

        self.cash_available = float(initial_cash)
        self.shares = int(initial_shares)
        self.tolerance = float(tolerance)
        self._receivables: dict[str, float] = {}
        self.cumulative_transaction_cost = 0.0
        self.cumulative_cash_interest = 0.0
        self.cumulative_dividends = 0.0
        self._assert_invariants()

    @property
    def dividend_receivable(self) -> float:
        """Total declared entitlement not yet transferred to spendable cash."""

        return float(sum(self._receivables.values()))

    @property
    def outstanding_dividend_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._receivables))

    def snapshot(self, true_price: float) -> LedgerSnapshot:
        """Value the account using an unperturbed market price."""

        self._validate_positive_price(true_price)
        market_value = self.shares * float(true_price)
        nav = self.cash_available + market_value + self.dividend_receivable
        self._validate_finite("nav", nav)
        if nav < -self.tolerance:
            raise RuntimeError("ledger NAV became negative")
        return LedgerSnapshot(
            true_price=float(true_price),
            cash_available=self.cash_available,
            shares=self.shares,
            dividend_receivable=self.dividend_receivable,
            market_value=market_value,
            nav=nav,
        )

    def accrue_cash(self, annual_yield_pct: float, calendar_days: int) -> float:
        """Accrue spendable cash with the frozen ACT/365 convention.

        ``annual_yield_pct`` is a percent value such as FRED DGS3MO's ``5.25``.
        Interest is applied only to available cash, never to a receivable.
        """

        self._validate_finite("annual_yield_pct", annual_yield_pct)
        if isinstance(calendar_days, bool) or not isinstance(calendar_days, int):
            raise TypeError("calendar_days must be an integer")
        if calendar_days < 0:
            raise ValueError("calendar_days must be non-negative")
        if annual_yield_pct <= -100:
            raise ValueError("annual_yield_pct must exceed -100")

        rate = (1.0 + float(annual_yield_pct) / 100.0) ** (
            calendar_days / 365.0
        ) - 1.0
        interest = self.cash_available * rate
        self.cash_available += interest
        self.cumulative_cash_interest += interest
        self._assert_invariants()
        return interest

    def create_dividend_receivable(
        self,
        event_id: str,
        *,
        per_share_amount: float,
        entitlement_shares: int,
    ) -> float:
        """Recognize an ex-date entitlement without making it spendable.

        The caller must pass the shares held at the frozen entitlement instant
        (pre-ex-date close).  Later trades therefore cannot change this amount.
        """

        if not event_id:
            raise ValueError("event_id must be non-empty")
        if event_id in self._receivables:
            raise ValueError(f"duplicate dividend event: {event_id}")
        self._validate_finite("per_share_amount", per_share_amount)
        self._validate_integer_shares(entitlement_shares)
        if per_share_amount < 0:
            raise ValueError("per_share_amount must be non-negative")

        amount = float(per_share_amount) * entitlement_shares
        self._receivables[event_id] = amount
        self.cumulative_dividends += amount
        self._assert_invariants()
        return amount

    def settle_dividend(self, event_id: str) -> float:
        """Move one payment-date receivable to spendable cash."""

        if event_id not in self._receivables:
            raise KeyError(f"unknown dividend event: {event_id}")
        amount = self._receivables.pop(event_id)
        self.cash_available += amount
        self._assert_invariants()
        return amount

    def apply_split(self, ratio: float) -> int:
        """Apply a split before open; reject unimplemented cash-in-lieu cases."""

        self._validate_finite("ratio", ratio)
        if ratio <= 0:
            raise ValueError("split ratio must be positive")
        post_shares = self.shares * float(ratio)
        rounded = round(post_shares)
        if not math.isclose(post_shares, rounded, rel_tol=0.0, abs_tol=self.tolerance):
            raise ValueError(
                "split would create fractional shares; cash-in-lieu is not implemented"
            )
        self.shares = int(rounded)
        self._assert_invariants()
        return self.shares

    def execute_target(
        self,
        target_weight: float,
        true_open: float,
        *,
        transaction_cost_bps: float = 10.0,
    ) -> TradeExecution:
        """Trade toward a target risky-asset weight using integer shares.

        Candidate shares are decremented until both trade notional and cost can
        be paid without borrowing.  This makes the no-negative-cash constraint
        explicit even at a 100% target weight.
        """

        self._validate_finite("target_weight", target_weight)
        self._validate_positive_price(true_open)
        self._validate_finite("transaction_cost_bps", transaction_cost_bps)
        if not 0.0 <= target_weight <= 1.0:
            raise ValueError("target_weight must be within [0, 1]")
        if transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative")

        pre = self.snapshot(true_open)
        cash_pre = self.cash_available
        shares_pre = self.shares
        cost_rate = float(transaction_cost_bps) / 10_000.0
        candidate = math.floor(float(target_weight) * pre.nav / float(true_open))

        while candidate >= 0:
            delta = candidate - shares_pre
            notional = delta * float(true_open)
            cost = cost_rate * abs(notional)
            cash_post = cash_pre - notional - cost
            if cash_post >= -self.tolerance:
                break
            candidate -= 1
        if candidate < 0:
            raise RuntimeError("no feasible non-negative share target")

        if abs(cash_post) <= self.tolerance:
            cash_post = 0.0
        self.cash_available = float(cash_post)
        self.shares = int(candidate)
        self.cumulative_transaction_cost += cost
        self._assert_invariants()

        post = self.snapshot(true_open)
        expected_post_nav = pre.nav - cost
        if not math.isclose(
            post.nav,
            expected_post_nav,
            rel_tol=1e-12,
            abs_tol=max(self.tolerance, 1e-8),
        ):
            raise RuntimeError("post-trade NAV identity failed")
        realized_weight = 0.0 if post.nav == 0 else post.market_value / post.nav

        return TradeExecution(
            target_weight=float(target_weight),
            realized_weight=realized_weight,
            shares_pre=shares_pre,
            shares_target=int(candidate),
            shares_post=self.shares,
            delta_shares=int(candidate - shares_pre),
            true_open=float(true_open),
            trade_notional=float(notional),
            transaction_cost=float(cost),
            cash_pre_trade=cash_pre,
            cash_post_trade=self.cash_available,
            nav_open_pre=pre.nav,
            nav_open_post=post.nav,
        )

    def hold_position(self, true_open: float) -> TradeExecution:
        """Keep the current integer share count without charging a trade cost."""

        snapshot = self.snapshot(true_open)
        realized_weight = (
            0.0 if snapshot.nav == 0 else snapshot.market_value / snapshot.nav
        )
        return TradeExecution(
            target_weight=realized_weight,
            realized_weight=realized_weight,
            shares_pre=self.shares,
            shares_target=self.shares,
            shares_post=self.shares,
            delta_shares=0,
            true_open=float(true_open),
            trade_notional=0.0,
            transaction_cost=0.0,
            cash_pre_trade=self.cash_available,
            cash_post_trade=self.cash_available,
            nav_open_pre=snapshot.nav,
            nav_open_post=snapshot.nav,
        )

    @staticmethod
    def open_to_open_reward(start_pretrade_nav: float, end_pretrade_nav: float) -> float:
        """Return the reward aligned to consecutive pre-trade open NAVs."""

        for name, value in (
            ("start_pretrade_nav", start_pretrade_nav),
            ("end_pretrade_nav", end_pretrade_nav),
        ):
            PortfolioLedger._validate_finite(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        return math.log(float(end_pretrade_nav) / float(start_pretrade_nav))

    def _assert_invariants(self) -> None:
        self._validate_finite("cash_available", self.cash_available)
        self._validate_integer_shares(self.shares)
        if self.cash_available < -self.tolerance:
            raise RuntimeError("ledger cash became negative")
        if abs(self.cash_available) <= self.tolerance:
            self.cash_available = 0.0
        for event_id, amount in self._receivables.items():
            if not event_id or not math.isfinite(amount) or amount < 0:
                raise RuntimeError("invalid dividend receivable state")

    @staticmethod
    def _validate_finite(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    @staticmethod
    def _validate_integer_shares(shares: int) -> None:
        if isinstance(shares, bool) or not isinstance(shares, int):
            raise TypeError("shares must be an integer")
        if shares < 0:
            raise ValueError("shares must be non-negative")

    @staticmethod
    def _validate_positive_price(price: float) -> None:
        PortfolioLedger._validate_finite("price", price)
        if price <= 0:
            raise ValueError("price must be positive")
