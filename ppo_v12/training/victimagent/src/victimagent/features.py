"""Production causal Observation Contract V1 feature builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from victimagent.data.splits import SplitContract
from victimagent.preprocessing import TrainOnlyZScore


MARKET_FEATURES = (
    "log_return_1",
    "log_return_5",
    "close_to_sma_5",
    "close_to_sma_20",
    "close_to_sma_60",
    "realized_volatility_20",
)
EXOGENOUS_FEATURES = ("cash_rate_known",)
CONTEXT_FEATURES = MARKET_FEATURES + EXOGENOUS_FEATURES
ACCOUNT_FEATURES = (
    "risky_weight_observed",
    "cash_weight_observed",
    "receivable_weight_observed",
)
OBSERVATION_FEATURES = CONTEXT_FEATURES + ACCOUNT_FEATURES
MAX_LOOKBACK_ROWS = 60


def canonical_market_subset(features: object) -> tuple[str, ...]:
    """Validate a market-feature subset and return it in canonical order.

    Ordering always follows ``MARKET_FEATURES`` so a candidate's observation
    layout is a deterministic function of *which* features it keeps, never of
    the order they were declared in. The full set therefore reproduces the
    frozen V1 observation exactly.
    """

    subset = tuple(str(f) for f in features)
    if not subset:
        raise ValueError("at least one market feature is required")
    unknown = [f for f in subset if f not in MARKET_FEATURES]
    if unknown:
        raise ValueError(f"unknown market features: {unknown}")
    if len(set(subset)) != len(subset):
        raise ValueError(f"duplicate market features: {subset}")
    return tuple(f for f in MARKET_FEATURES if f in set(subset))


@dataclass(frozen=True)
class AccountState:
    cash_available: float
    shares: int
    dividend_receivable: float = 0.0

    def validate(self) -> None:
        values = (self.cash_available, float(self.shares), self.dividend_receivable)
        if not all(np.isfinite(values)):
            raise ValueError("account state must be finite")
        if self.cash_available < -1e-8:
            raise ValueError("cash_available cannot be negative")
        if not isinstance(self.shares, (int, np.integer)) or self.shares < 0:
            raise ValueError("shares must be a non-negative integer")
        if self.dividend_receivable < 0:
            raise ValueError("dividend_receivable cannot be negative")


def build_context_features(
    frame: pd.DataFrame,
    *,
    price_basis: str = "causal_total_return",
) -> pd.DataFrame:
    """Build the seven causal market/exogenous features for one asset.

    The function intentionally returns warm-up NaNs. Callers decide which
    dates are eligible; filling a missing feature is forbidden by contract.
    """

    required = {"date", "close_raw", "ordinary_dividend", "dgs3mo_pct"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing production feature inputs: {sorted(missing)}")
    for identity in ("permno", "ticker"):
        if identity in frame.columns and frame[identity].nunique(dropna=False) != 1:
            raise ValueError("build_context_features accepts exactly one asset")

    result = frame.copy().sort_values("date").reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"])
    if result["date"].duplicated().any() or not result["date"].is_monotonic_increasing:
        raise ValueError("feature input dates must be unique and increasing")

    close = pd.to_numeric(result["close_raw"], errors="coerce")
    if close.isna().any() or (~np.isfinite(close)).any() or (close <= 0).any():
        raise ValueError("close_raw must be finite and strictly positive")
    dividend = pd.to_numeric(result["ordinary_dividend"], errors="coerce")
    if dividend.isna().any() or (~np.isfinite(dividend)).any() or (dividend < 0).any():
        raise ValueError("ordinary_dividend must be finite and non-negative")
    if price_basis == "causal_total_return":
        factors = (close + dividend) / close.shift(1)
        factors.iloc[0] = 1.0
        if (~np.isfinite(factors)).any() or (factors <= 0).any():
            raise ValueError("causal total-return factors must be finite and positive")
        feature_price = 100.0 * factors.cumprod()
    elif price_basis == "raw":
        feature_price = close.copy()
    else:
        raise ValueError("price_basis must be 'causal_total_return' or 'raw'")
    result["causal_total_return_index"] = 100.0 * (
        ((close + dividend) / close.shift(1)).fillna(1.0).cumprod()
    )
    result["feature_price_basis"] = price_basis
    result["feature_price"] = feature_price
    log_close = np.log(feature_price)
    one_day = log_close.diff()

    result["log_return_1"] = one_day
    result["log_return_5"] = log_close - log_close.shift(5)
    for window in (5, 20, 60):
        sma = feature_price.rolling(window=window, min_periods=window).mean()
        result[f"feature_sma_{window}"] = sma
        result[f"close_to_sma_{window}"] = feature_price / sma - 1.0
    result["realized_volatility_20"] = (
        one_day.rolling(window=20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    )
    result["cash_rate_known"] = pd.to_numeric(
        result["dgs3mo_pct"], errors="coerce"
    ) / 100.0
    return result


class ObservationBuilderV1:
    """Fit train-only scaling and compose the causal observation.

    The default reproduces the frozen V1 10-D observation exactly (all six
    market features, the known cash rate, and the three account fractions). The
    V2 ``observation_ablation`` stage passes a market-feature subset and/or drops
    the cash rate; the account fractions are structural and always present. The
    scaler, fit guard, ordering and NAV/finiteness checks are otherwise
    unchanged, so an ablated builder is the same engine over fewer columns.
    """

    def __init__(
        self,
        market_features: object | None = None,
        *,
        include_cash_rate: bool = True,
    ) -> None:
        if market_features is None:
            self.market_features: tuple[str, ...] = MARKET_FEATURES
        else:
            self.market_features = canonical_market_subset(market_features)
        self.include_cash_rate = bool(include_cash_rate)
        self.context_features: tuple[str, ...] = self.market_features + (
            EXOGENOUS_FEATURES if self.include_cash_rate else ()
        )
        self.observation_features: tuple[str, ...] = (
            self.context_features + ACCOUNT_FEATURES
        )
        self.scaler = TrainOnlyZScore(self.context_features)

    def fit(
        self,
        feature_frame: pd.DataFrame,
        contract: SplitContract,
    ) -> "ObservationBuilderV1":
        train = contract.select(feature_frame, "train")
        columns = list(self.context_features)
        if train.loc[:, columns].isna().any().any():
            raise ValueError("training features contain missing values")
        self.scaler.fit(train[["date", *columns]], contract)
        return self

    def transform_context(
        self,
        feature_frame: pd.DataFrame,
        *,
        require_complete: bool = True,
    ) -> pd.DataFrame:
        columns = list(self.context_features)
        transformed = self.scaler.transform(feature_frame)
        if require_complete and transformed.loc[:, columns].isna().any().any():
            raise ValueError("eligible context features contain missing values")
        values = transformed.loc[:, columns].to_numpy(dtype=np.float64)
        if require_complete and not np.isfinite(values).all():
            raise ValueError("eligible context features must be finite")
        transformed.loc[:, columns] = transformed.loc[:, columns].astype(np.float32)
        return transformed

    def scaler_state(self) -> dict[str, dict[str, float]]:
        if self.scaler.mean_ is None or self.scaler.scale_ is None:
            raise RuntimeError("fit must be called before scaler_state")
        return {
            "mean": {k: float(v) for k, v in self.scaler.mean_.items()},
            "scale": {k: float(v) for k, v in self.scaler.scale_.items()},
        }

    def compose(
        self,
        scaled_context: Mapping[str, float] | pd.Series,
        account: AccountState,
        *,
        observed_close: float,
    ) -> np.ndarray:
        account.validate()
        if not np.isfinite(observed_close) or observed_close <= 0:
            raise ValueError("observed_close must be finite and positive")

        context = np.asarray(
            [float(scaled_context[name]) for name in self.context_features],
            dtype=np.float64,
        )
        perceived_nav = (
            account.cash_available
            + account.shares * observed_close
            + account.dividend_receivable
        )
        if not np.isfinite(perceived_nav) or perceived_nav <= 0:
            raise ValueError("perceived NAV must be finite and positive")
        account_features = np.asarray(
            [
                account.shares * observed_close / perceived_nav,
                account.cash_available / perceived_nav,
                account.dividend_receivable / perceived_nav,
            ],
            dtype=np.float64,
        )
        observation = np.concatenate([context, account_features])
        if observation.shape != (len(self.observation_features),):
            raise AssertionError("observation shape violates the builder contract")
        if not np.isfinite(observation).all():
            raise ValueError("observation must be finite")
        if not np.isclose(account_features.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise AssertionError("account weights must sum to one")
        return observation.astype(np.float32)
