"""Adapter from frozen CRSP/FRED pipeline outputs to Environment V1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from victimagent.data.splits import SplitContract
from victimagent.environment import DividendEvent, SplitEvent
from victimagent.features import (
    ObservationBuilderV1,
    build_context_features,
)


@dataclass(frozen=True)
class RealDataBundle:
    ticker: str
    permno: int
    market_true: pd.DataFrame
    market_obs: pd.DataFrame
    observation_builder: ObservationBuilderV1
    dividend_events: tuple[DividendEvent, ...]
    split_events: tuple[SplitEvent, ...]
    metadata: dict[str, int | str]


def prepare_asset_bundle(
    processed_market: pd.DataFrame,
    distributions: pd.DataFrame,
    contract: SplitContract,
    *,
    ticker: str,
    partition: str = "formal",
    market_features: object | None = None,
    include_cash_rate: bool = True,
) -> RealDataBundle:
    """Create fully aligned true/observed frames for one frozen asset.

    ``market_features``/``include_cash_rate`` select the observation layout. The
    default (``None``, ``True``) reproduces the frozen V1 10-D observation; the
    V2 ``observation_ablation`` stage passes a market-feature subset. Only the
    builder's own context columns are scaled and carried into ``market_obs``, so
    each layout gets its own train-only scaler over exactly the features it uses.
    """

    ticker = ticker.upper()
    required_market = {
        "date",
        "permno",
        "ticker",
        "open_raw",
        "close_raw",
        "dgs3mo_pct",
        "cumulative_share_factor",
    }
    missing = required_market - set(processed_market.columns)
    if missing:
        raise KeyError(f"processed market missing columns: {sorted(missing)}")

    asset = processed_market.loc[
        processed_market["ticker"].astype(str).str.upper() == ticker
    ].copy()
    if asset.empty:
        raise ValueError(f"ticker not found in processed market: {ticker}")
    if asset["permno"].nunique() != 1:
        raise ValueError(f"{ticker} must map to exactly one PERMNO")
    asset["date"] = pd.to_datetime(asset["date"]).dt.normalize()
    asset = asset.sort_values("date").reset_index(drop=True)
    if asset["date"].duplicated().any():
        raise ValueError(f"{ticker} has duplicate market dates")

    share_factor = pd.to_numeric(asset["cumulative_share_factor"], errors="coerce")
    if share_factor.isna().any() or not np.isfinite(share_factor).all():
        raise ValueError("cumulative_share_factor must be finite")
    if not np.allclose(share_factor.to_numpy(), share_factor.iloc[0]):
        raise NotImplementedError(
            "real split-factor mapping is not frozen; create explicit SplitEvent rows first"
        )

    features = build_context_features(asset)
    builder = ObservationBuilderV1(
        market_features=market_features, include_cash_rate=include_cash_rate
    ).fit(features, contract)
    context_features = list(builder.context_features)
    scaled = builder.transform_context(features, require_complete=False)
    mask = _partition_mask(scaled["date"], contract, partition)
    selected = scaled.loc[mask].copy().reset_index(drop=True)
    if len(selected) < 3:
        raise ValueError(f"partition {partition!r} has fewer than three rows")
    values = selected.loc[:, context_features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("selected environment rows contain incomplete context features")

    market_true = selected[
        ["date", "open_raw", "close_raw", "dgs3mo_pct"]
    ].copy()
    market_obs = selected[["date", *context_features]].copy()
    market_obs.insert(1, "observed_close", selected["close_raw"].astype(float))
    market_obs["ma_fast_above_slow"] = (
        selected["feature_sma_20"] > selected["feature_sma_60"]
    ).astype(np.int8)

    dividends, mapped_payments, settled_payments = _build_dividends(
        distributions,
        ticker=ticker,
        sessions=pd.DatetimeIndex(market_true["date"]),
    )
    permno = int(asset["permno"].iloc[0])
    return RealDataBundle(
        ticker=ticker,
        permno=permno,
        market_true=market_true,
        market_obs=market_obs,
        observation_builder=builder,
        dividend_events=tuple(dividends),
        split_events=(),
        metadata={
            "partition": partition,
            "rows": len(market_true),
            "first_date": str(market_true["date"].iloc[0].date()),
            "last_date": str(market_true["date"].iloc[-1].date()),
            "dividend_events": len(dividends),
            # Payments whose CRSP pay-date lands on a trading session inside the
            # partition (directly or after remapping): these are actually
            # settled to cash during a rollout.
            "payments_settled_in_partition": settled_payments,
            # Subset of the above whose raw pay-date was a non-session and had to
            # be remapped forward to the next session (0 is normal, not a defect).
            "payment_dates_mapped_to_next_session": mapped_payments,
            # Receivables created on an ex-date but whose pay-date falls past the
            # partition's last session; they stay in NAV but never reach cash.
            "receivables_outstanding_at_boundary": len(dividends) - settled_payments,
            "split_events": 0,
        },
    )


def _partition_mask(
    dates: pd.Series, contract: SplitContract, partition: str
) -> pd.Series:
    dates = pd.to_datetime(dates)
    if partition == "formal":
        return (dates >= contract.eligible_start) & (dates <= contract.eligible_end)
    selected = contract.range_for(partition)
    return selected.contains(dates)


def _build_dividends(
    distributions: pd.DataFrame,
    *,
    ticker: str,
    sessions: pd.DatetimeIndex,
) -> tuple[list[DividendEvent], int, int]:
    required = {
        "ticker",
        "disordinaryflg",
        "disdivamt",
        "disexdt",
        "dispaydt",
    }
    missing = required - set(distributions.columns)
    if missing:
        raise KeyError(f"distribution data missing columns: {sorted(missing)}")

    rows = distributions.loc[
        (distributions["ticker"].astype(str).str.upper() == ticker)
        & (distributions["disordinaryflg"].astype(str).str.upper() == "Y")
    ].copy()
    if rows.empty:
        return [], 0, 0
    rows["disexdt"] = pd.to_datetime(rows["disexdt"]).dt.normalize()
    rows["dispaydt"] = pd.to_datetime(rows["dispaydt"]).dt.normalize()
    rows["disdivamt"] = pd.to_numeric(rows["disdivamt"], errors="coerce")
    if rows[["disexdt", "dispaydt", "disdivamt"]].isna().any().any():
        raise ValueError("ordinary dividend rows contain missing required values")
    if (rows["disdivamt"] < 0).any():
        raise ValueError("ordinary dividend amount cannot be negative")

    session_values = sessions.sort_values().unique()
    session_set = set(session_values)
    first, last = session_values[0], session_values[-1]
    rows = rows.loc[rows["disexdt"].isin(session_set)].copy()
    events: list[DividendEvent] = []
    mapped = 0
    settled = 0
    for ordinal, (_, row) in enumerate(rows.sort_values("disexdt").iterrows(), start=1):
        pay = pd.Timestamp(row["dispaydt"])
        mapped_pay = pay
        if pay in session_set:
            settled += 1
        elif first <= pay <= last:
            candidates = session_values[session_values >= pay]
            if len(candidates) == 0:
                raise ValueError("payment date cannot be mapped within the selected calendar")
            mapped_pay = pd.Timestamp(candidates[0])
            mapped += 1
            settled += 1
        source_key = row.get("source_row_number", ordinal)
        event_id = f"{ticker}-{pd.Timestamp(row['disexdt']):%Y%m%d}-{source_key}"
        events.append(
            DividendEvent(
                event_id=event_id,
                ex_date=pd.Timestamp(row["disexdt"]),
                pay_date=mapped_pay,
                per_share_amount=float(row["disdivamt"]),
            )
        )
    return events, mapped, settled
