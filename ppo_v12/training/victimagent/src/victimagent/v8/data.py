"""V8 SPY data splice: 1993-2008 + frozen 2009-2025 CRSP/FRED inputs.

Raw files are read, never modified. Every irregular raw row must be named in
``configs/v8_final.yaml``; anything unlisted fails the build instead of being
silently filled, dropped or reclassified.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PRICE_COLUMNS = ("dlyopen", "dlyhigh", "dlylow", "dlyclose")
SESSION_ACTIONS = frozenset({"exclude_session", "warmup_only_missing_open"})


def load_v8_config(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / "configs/v8_final.yaml").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _complete_ohlc(frame: pd.DataFrame) -> pd.Series:
    prices = frame.loc[:, list(PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return prices.notna().all(axis=1) & (prices > 0).all(axis=1)


def splice_market(frames: list[pd.DataFrame], *, permno: int, ticker: str,
                  exceptions: dict[str, str], train_start: pd.Timestamp,
                  seam: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Concatenate eras for one asset, apply only the declared exceptions."""
    columns = list(frames[0].columns)
    for frame in frames[1:]:
        if list(frame.columns) != columns:
            raise ValueError("market files must share an identical CRSP schema")
    parts = []
    for frame in frames:
        part = frame.loc[pd.to_numeric(frame["permno"], errors="coerce") == permno].copy()
        if part.empty:
            raise ValueError(f"a market file has no rows for PERMNO {permno}")
        if set(part["ticker"].astype(str).str.upper()) != {ticker}:
            raise ValueError(f"PERMNO {permno} does not map only to {ticker}")
        parts.append(part)
    market = pd.concat(parts, ignore_index=True)
    market["date"] = pd.to_datetime(market["dlycaldt"], errors="raise").dt.normalize()
    if market["date"].duplicated().any():
        raise ValueError("market eras overlap: duplicate SPY dates")
    market = market.sort_values("date").reset_index(drop=True)
    for i in range(1, len(parts)):
        if pd.to_datetime(parts[i]["dlycaldt"]).min() <= pd.to_datetime(parts[i - 1]["dlycaldt"]).max():
            raise ValueError("market files must be supplied in chronological order")

    bad_status = (market["tradingstatusflg"] != "A") | (market["conditionaltype"] != "RW")
    if bad_status.any():
        raise ValueError(f"non-regular CRSP rows: {market.loc[bad_status, 'date'].dt.date.tolist()}")

    complete = _complete_ohlc(market)
    close_ok = pd.to_numeric(market["dlyclose"], errors="coerce").gt(0)
    irregular = {str(d.date()) for d in market.loc[~complete, "date"]}
    declared = set(exceptions)
    unknown_actions = {v for v in exceptions.values() if v not in SESSION_ACTIONS}
    if unknown_actions:
        raise ValueError(f"unknown session exception actions: {sorted(unknown_actions)}")
    if irregular != declared:
        raise ValueError(
            "irregular raw rows must equal the declared exceptions: "
            f"undeclared={sorted(irregular - declared)}, stale={sorted(declared - irregular)}"
        )
    keep = pd.Series(True, index=market.index)
    for day, action in exceptions.items():
        mask = market["date"] == pd.Timestamp(day)
        if action == "exclude_session":
            keep &= ~mask
        else:
            if pd.Timestamp(day) >= train_start:
                raise ValueError(f"{day}: warm-up-only row lies inside a training window")
            if not close_ok[mask].all():
                raise ValueError(f"{day}: warm-up-only row still needs a valid close")
    market = market.loc[keep].reset_index(drop=True)

    ohlc = market.loc[:, list(PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    bounds_bad = ((ohlc["dlylow"] > ohlc["dlyopen"]) | (ohlc["dlylow"] > ohlc["dlyclose"]) |
                  (ohlc["dlyopen"] > ohlc["dlyhigh"]) | (ohlc["dlyclose"] > ohlc["dlyhigh"]))
    if bounds_bad.any():
        raise ValueError("OHLC bounds violated")
    factor = pd.to_numeric(market["dlycumfacshr"], errors="coerce")
    if not np.allclose(factor, factor.iloc[0]):
        raise NotImplementedError("SPY share factor changed; explicit split events required")

    last_early = pd.Timestamp(seam["last_early_session"])
    first_late = pd.Timestamp(seam["first_late_session"])
    early = market.loc[market["date"] == last_early]
    late = market.loc[market["date"] == first_late]
    if len(early) != 1 or len(late) != 1:
        raise ValueError("seam sessions are missing")
    idx = late.index[0]
    if idx != early.index[0] + 1:
        raise ValueError("seam sessions are not adjacent")
    implied = (float(late["dlyclose"].iloc[0]) + float(late["dlyorddivamt"].fillna(0).iloc[0])
               ) / float(early["dlyclose"].iloc[0]) - 1.0
    seam_error = abs(implied - float(late["dlyret"].iloc[0]))
    if seam_error > float(seam["return_tolerance"]):
        raise ValueError(f"era seam return mismatch: {seam_error}")

    # CRSP's own total return must agree with close + cash distributions for
    # every retained adjacent pair; a gap left by an excluded session is
    # checked against the compounded return.
    cash = market["dlyorddivamt"].fillna(0.0) + market["dlynonorddivamt"].fillna(0.0)
    close = pd.to_numeric(market["dlyclose"])
    implied_all = (close + cash) / close.shift(1) - 1.0
    crsp = pd.to_numeric(market["dlyret"], errors="coerce")
    excluded = [pd.Timestamp(d) for d, a in exceptions.items() if a == "exclude_session"]
    gap_rows = market["date"].isin(
        [market.loc[market["date"] > d, "date"].min() for d in excluded])
    check = implied_all.notna() & crsp.notna() & ~gap_rows
    worst = float((implied_all[check] - crsp[check]).abs().max())
    return market, {"rows": int(len(market)), "seam_return_error": seam_error,
                    "max_abs_crsp_return_reconciliation_error": worst,
                    "first_date": str(market["date"].iloc[0].date()),
                    "last_date": str(market["date"].iloc[-1].date())}


def cash_distributions(frames: list[pd.DataFrame], *, permno: int, ticker: str,
                       allowlist: list[dict[str, str]]) -> pd.DataFrame:
    rows = []
    for frame in frames:
        part = frame.loc[pd.to_numeric(frame["permno"], errors="coerce") == permno].copy()
        rows.append(part)
    events = pd.concat(rows, ignore_index=True)
    allowed = {(a["distype"], a["disdetailtype"], a["disordinaryflg"]) for a in allowlist}
    keys = list(zip(events["distype"].astype(str), events["disdetailtype"].astype(str),
                    events["disordinaryflg"].astype(str)))
    unknown = sorted({k for k in keys if k not in allowed})
    if unknown:
        raise ValueError(f"distribution types outside the cash allowlist: {unknown}")
    for column in ("disexdt", "dispaydt"):
        events[column] = pd.to_datetime(events[column], errors="raise").dt.normalize()
    events["disdivamt"] = pd.to_numeric(events["disdivamt"], errors="raise")
    if events[["disexdt", "dispaydt", "disdivamt"]].isna().any().any():
        raise ValueError("cash distributions need ex-date, pay-date and amount")
    if (events["disdivamt"] < 0).any() or (events["dispaydt"] < events["disexdt"]).any():
        raise ValueError("invalid cash distribution amount or dates")
    if events.duplicated(["disexdt", "disseqnbr"]).any():
        raise ValueError("duplicate distribution event keys")
    events["ticker"] = ticker
    events["event_id"] = [f"{ticker}-{d:%Y%m%d}-{int(s)}"
                          for d, s in zip(events["disexdt"], events["disseqnbr"])]
    return events.sort_values(["disexdt", "disseqnbr"]).reset_index(drop=True)


def reconcile(market: pd.DataFrame, events: pd.DataFrame, tolerance: float) -> dict[str, Any]:
    daily = market.assign(amount=market["dlyorddivamt"].fillna(0.0) +
                          market["dlynonorddivamt"].fillna(0.0))
    daily = daily.loc[daily["amount"] != 0, ["date", "amount"]]
    by_ex = events.groupby("disexdt", as_index=False)["disdivamt"].sum()
    merged = daily.merge(by_ex, left_on="date", right_on="disexdt", how="outer", indicator=True)
    daily_only = merged.loc[merged["_merge"] == "left_only", "date"].dt.date.astype(str).tolist()
    events_only = merged.loc[merged["_merge"] == "right_only", "disexdt"].dt.date.astype(str).tolist()
    both = merged["_merge"] == "both"
    mismatch = (merged.loc[both, "amount"] - merged.loc[both, "disdivamt"]).abs() > tolerance
    if daily_only or events_only or mismatch.any():
        raise ValueError(f"distribution reconciliation failed: daily_only={daily_only}, "
                         f"events_only={events_only}, amount_mismatch={int(mismatch.sum())}")
    return {"matched_ex_dates": int(both.sum()), "events": int(len(events))}


def read_cash(frames: list[pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for frame in frames:
        x = frame.rename(columns={"DATE": "observation_date"}).copy()
        if "observation_date" not in x or "DGS3MO" not in x:
            raise KeyError("cash file needs observation_date/DATE and DGS3MO")
        parts.append(pd.DataFrame({
            "observation_date": pd.to_datetime(x["observation_date"], errors="raise"),
            "dgs3mo_pct": pd.to_numeric(x["DGS3MO"], errors="coerce"),
        }))
    rates = pd.concat(parts, ignore_index=True)
    duplicated = rates["observation_date"].duplicated(keep=False)
    if duplicated.any():
        groups = rates.loc[duplicated].groupby("observation_date")["dgs3mo_pct"]
        if (groups.nunique(dropna=False) > 1).any():
            raise ValueError("overlapping DGS3MO files disagree")
        rates = rates.drop_duplicates("observation_date")
    return rates.sort_values("observation_date").reset_index(drop=True)


def build_v8_data(root: Path, data_root: Path | None = None) -> dict[str, Any]:
    data_root = data_root or root
    cfg = load_v8_config(root)
    data = cfg["data"]
    raw = data_root / data["raw_dir"]
    names = [*data["market_files"], *data["distribution_files"], *data["cash_files"]]
    hashes = {name: sha256(raw / name) for name in names}
    for name, expected in data["pinned_sha256"].items():
        if hashes.get(name) != expected:
            raise ValueError(f"{name}: sha256 {hashes.get(name)} != pinned {expected}")

    ticker, permno = str(cfg["asset"]).upper(), int(cfg["permno"])
    train_start = pd.Timestamp(cfg["final"]["v8_long_horizon"]["train_start"])
    market, market_audit = splice_market(
        [pd.read_csv(raw / n) for n in data["market_files"]], permno=permno, ticker=ticker,
        exceptions={str(k): v for k, v in data["session_exceptions"].items()},
        train_start=train_start, seam=data["seam"])
    events = cash_distributions([pd.read_csv(raw / n) for n in data["distribution_files"]],
                                permno=permno, ticker=ticker,
                                allowlist=data["cash_distribution_allowlist"])
    sessions = set(market["date"])
    off_calendar = events.loc[~events["disexdt"].isin(sessions), "disexdt"]
    if not off_calendar.empty:
        raise ValueError(f"ex-dates missing from the session calendar: {off_calendar.tolist()}")
    dividend_audit = reconcile(market, events, float(data["dividend_amount_tolerance"]))

    rates = read_cash([pd.read_csv(raw / n) for n in data["cash_files"]])
    valid = rates.dropna(subset=["dgs3mo_pct"])
    lagged = pd.merge_asof(market[["date"]], valid, left_on="date", right_on="observation_date",
                           direction="backward", allow_exact_matches=False)
    if lagged["dgs3mo_pct"].isna().any():
        raise ValueError("lagged DGS3MO missing for some sessions")
    stale = (lagged["date"] - lagged["observation_date"]).dt.days
    if stale.max() > 10:
        raise ValueError(f"lagged DGS3MO is stale by {int(stale.max())} days")

    out = pd.DataFrame({
        "date": market["date"].dt.strftime("%Y-%m-%d"),
        "permno": permno, "ticker": ticker,
        "open_raw": pd.to_numeric(market["dlyopen"], errors="coerce"),
        "high_raw": pd.to_numeric(market["dlyhigh"], errors="coerce"),
        "low_raw": pd.to_numeric(market["dlylow"], errors="coerce"),
        "close_raw": pd.to_numeric(market["dlyclose"]),
        "volume_raw": market["dlyvol"],
        "cash_distribution": market["dlyorddivamt"].fillna(0.0) + market["dlynonorddivamt"].fillna(0.0),
        "total_return_audit_only": market["dlyret"],
        "cumulative_share_factor": market["dlycumfacshr"],
        "dgs3mo_pct": lagged["dgs3mo_pct"].to_numpy(),
        "dgs3mo_source_date": lagged["observation_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "open_tradable": _complete_ohlc(market).to_numpy(),
    })
    ev = events[["event_id", "ticker", "permno", "disexdt", "dispaydt", "disdivamt", "distype",
                 "disdetailtype", "disordinaryflg", "disseqnbr"]].copy()
    for column in ("disexdt", "dispaydt"):
        ev[column] = ev[column].dt.strftime("%Y-%m-%d")

    market_path = data_root / data["processed_market"]
    events_path = data_root / data["processed_events"]
    for path in (market_path, events_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(market_path, index=False)
    ev.to_csv(events_path, index=False)
    report = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": hashes,
        "processed_sha256": {data["processed_market"]: sha256(market_path),
                             data["processed_events"]: sha256(events_path)},
        "market": market_audit, "distributions": dividend_audit,
        "distribution_types": events.groupby(["distype", "disdetailtype", "disordinaryflg"]).size()
                                    .rename("count").reset_index().to_dict("records"),
        "session_exceptions": data["session_exceptions"],
        "max_cash_rate_staleness_days": int(stale.max()),
    }
    report_path = root / data["build_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report
