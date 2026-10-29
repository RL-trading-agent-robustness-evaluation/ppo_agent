"""V8 long-horizon, price-only causal features and observation builder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from victimagent.data.splits import DateRange, SplitContract
from victimagent.environment import DividendEvent
from victimagent.features import ACCOUNT_FEATURES, EXOGENOUS_FEATURES, ObservationBuilderV1
from victimagent.preprocessing import TrainOnlyZScore

V1_MARKET_FEATURES = (
    "log_return_1", "log_return_5", "close_to_sma_5", "close_to_sma_20",
    "close_to_sma_60", "realized_volatility_20",
)
LONG_HORIZON_FEATURES = (
    "close_to_sma_200", "log_return_126", "log_return_252",
    "realized_volatility_60", "volatility_ratio_20_60",
)
V8_MARKET_FEATURES = V1_MARKET_FEATURES + LONG_HORIZON_FEATURES
V8_CONTEXT_FEATURES = V8_MARKET_FEATURES + EXOGENOUS_FEATURES
V8_WARMUP_ROWS = 252


def build_v8_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Every value at row t uses closes/distributions at rows <= t only.

    The total-return index includes every allowlisted cash distribution on its
    ex-date (ordinary, special and capital-gain cash), so an ex-date price drop
    is not mistaken for a market move. Warm-up rows stay NaN; nothing is filled.
    """
    required = {"date", "close_raw", "cash_distribution", "dgs3mo_pct"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing V8 feature inputs: {sorted(missing)}")
    out = frame.copy().sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    if out["date"].duplicated().any():
        raise ValueError("V8 feature dates must be unique")
    close = pd.to_numeric(out["close_raw"], errors="raise")
    cash = pd.to_numeric(out["cash_distribution"], errors="raise")
    if (close <= 0).any() or not np.isfinite(close).all() or (cash < 0).any():
        raise ValueError("close must be positive and distributions non-negative")
    factor = ((close + cash) / close.shift(1)).fillna(1.0)
    price = 100.0 * factor.cumprod()
    out["feature_price"] = price
    log_price = np.log(price)
    one_day = log_price.diff()
    out["log_return_1"] = one_day
    for window in (5, 126, 252):
        out[f"log_return_{window}"] = log_price - log_price.shift(window)
    for window in (5, 20, 60, 200):
        sma = price.rolling(window, min_periods=window).mean()
        out[f"feature_sma_{window}"] = sma
        out[f"close_to_sma_{window}"] = price / sma - 1.0
    for window in (20, 60):
        out[f"realized_volatility_{window}"] = (
            one_day.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252.0))
    out["volatility_ratio_20_60"] = out["realized_volatility_20"] / out["realized_volatility_60"]
    out["cash_rate_known"] = pd.to_numeric(out["dgs3mo_pct"], errors="coerce") / 100.0
    return out


class ObservationBuilderV8(ObservationBuilderV1):
    """Same scaler guard / account composition as V1 over the V8 columns."""

    def __init__(self) -> None:  # noqa: D107 - layout is fixed by contract
        self.market_features = V8_MARKET_FEATURES
        self.include_cash_rate = True
        self.context_features = V8_CONTEXT_FEATURES
        self.observation_features = V8_CONTEXT_FEATURES + ACCOUNT_FEATURES
        self.scaler = TrainOnlyZScore(self.context_features)


@dataclass(frozen=True)
class V8Bundle:
    market_true: pd.DataFrame
    market_obs: pd.DataFrame
    observation_builder: ObservationBuilderV8
    dividend_events: tuple[DividendEvent, ...]
    split_events: tuple = ()
    metadata: dict | None = None


def _fit_contract(train: DateRange) -> SplitContract:
    # Only ``train`` is consulted by the train-only scaler guard.
    return SplitContract(train=train, validation=train, test=train,
                         eligible_start=train.start, eligible_end=train.end)


def prepare_v8_bundle(market: pd.DataFrame, events: pd.DataFrame, *,
                      fit: DateRange, select: DateRange) -> V8Bundle:
    """Fit the scaler on ``fit`` sessions; return env frames for ``select``."""
    features = build_v8_features(market)
    builder = ObservationBuilderV8().fit(features, _fit_contract(fit))
    scaled = builder.transform_context(features, require_complete=False)
    rows = scaled.loc[select.contains(scaled["date"])].reset_index(drop=True)
    if len(rows) < 3:
        raise ValueError("selected V8 window has fewer than three sessions")
    context = list(builder.context_features)
    if not np.isfinite(rows[context].to_numpy(dtype=float)).all():
        raise ValueError("selected V8 rows have incomplete features (warm-up too short)")
    if "open_tradable" in rows and not rows["open_tradable"].astype(bool).all():
        raise ValueError("selected V8 window contains a non-tradable open")
    market_true = rows[["date", "open_raw", "close_raw", "dgs3mo_pct"]].copy()
    market_obs = rows[["date", *context]].copy()
    market_obs.insert(1, "observed_close", rows["close_raw"].astype(float))
    market_obs["ma_fast_above_slow"] = (rows["feature_sma_20"] > rows["feature_sma_60"]).astype(np.int8)

    sessions = pd.DatetimeIndex(market_true["date"])
    session_values = sessions.values
    ev = events.copy()
    ev["disexdt"] = pd.to_datetime(ev["disexdt"]).dt.normalize()
    ev["dispaydt"] = pd.to_datetime(ev["dispaydt"]).dt.normalize()
    ev = ev.loc[ev["disexdt"].isin(set(sessions))].sort_values(["disexdt", "event_id"])
    dividends = []
    for _, row in ev.iterrows():
        pay = pd.Timestamp(row["dispaydt"])
        if sessions[0] <= pay <= sessions[-1] and pay not in set(sessions):
            pay = pd.Timestamp(session_values[session_values >= np.datetime64(pay)][0])
        dividends.append(DividendEvent(event_id=str(row["event_id"]), ex_date=row["disexdt"],
                                       pay_date=pay, per_share_amount=float(row["disdivamt"])))
    return V8Bundle(market_true=market_true, market_obs=market_obs, observation_builder=builder,
                    dividend_events=tuple(dividends),
                    metadata={"rows": len(market_true),
                              "first_date": str(sessions[0].date()),
                              "last_date": str(sessions[-1].date()),
                              "cash_distribution_events": len(dividends)})
