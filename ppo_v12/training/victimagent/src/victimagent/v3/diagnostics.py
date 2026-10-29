"""Read-only, fold/seed paired V3 diagnostic; it never selects a policy."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math

import numpy as np
import pandas as pd

from victimagent.baselines import run_baseline
from victimagent.data.real_adapter import prepare_asset_bundle
from victimagent.v2.folds import audit_folds, load_rolling_folds
from victimagent.v2.registry import sha256_file
from victimagent.v2.run import _development_only
from victimagent.v2.select import validate_stage_completeness
from victimagent.v3.run import CONTROL, FOLDS, _expected_ids, _records, load_contract


def audit_trajectory(root: Path, row: dict) -> dict:
    base = (root / "artifacts/trajectories/v3").resolve()
    path = (root / row["trajectory_path"]).resolve()
    if not path.is_relative_to(base) or "_validation" not in path.stem or path.suffix != ".csv":
        raise ValueError("V3 trajectory path must stay in the V3 validation directory")
    if sha256_file(path) != row["trajectory_sha256"]:
        raise ValueError("V3 validation trajectory hash mismatch")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("empty V3 trajectory")
    required = ("turnover", "transaction_cost", "trade_notional", "start_nav",
                "gross_return", "net_return", "cash_benchmark_return",
                "execution_date", "valuation_date")
    values = frame[list(required[:7])].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 3] <= 0).any():
        raise ValueError("non-finite or nonpositive V3 ledger value")
    turnover, cost, notional, nav, gross, net, cash = values.T
    if not np.allclose(turnover, np.abs(notional) / nav, atol=1e-9, rtol=1e-7):
        raise ValueError("turnover does not match V3 notional / NAV")
    if not np.allclose(gross - net, cost / nav, atol=1e-9, rtol=1e-7):
        raise ValueError("V3 gross/net cost does not reconcile")
    if not math.isclose(float(turnover.sum()), row["total_turnover"], rel_tol=1e-7, abs_tol=1e-6):
        raise ValueError("turnover does not reconcile with V3 registry")
    if not math.isclose(float(cost.sum()), row["total_transaction_cost"], rel_tol=1e-7, abs_tol=1e-6):
        raise ValueError("cost does not reconcile with V3 registry")
    starts = pd.to_datetime(frame["execution_date"], errors="raise")
    ends = pd.to_datetime(frame["valuation_date"], errors="raise")
    if starts.duplicated().any() or not starts.is_monotonic_increasing or (ends < starts).any():
        raise ValueError("invalid V3 trajectory timeline")
    cause_totals = {name: 0.0 for name in ("initial_entry", "action_change", "same_action", "skipped_order")}
    previous = None
    for ix, item in frame.iterrows():
        placed = str(item["order_placed"]).lower() == "true"
        if placed:
            action = int(item["executed_action"])
            cause = ("initial_entry" if previous is None else
                     "action_change" if action != previous else "same_action")
            previous = action
        else:
            cause = "skipped_order"
            if not math.isclose(float(turnover[ix]), 0.0, abs_tol=1e-12):
                raise ValueError("held V3 order has nonzero turnover")
        cause_totals[cause] += float(turnover[ix])
    if any(not math.isclose(value, row["turnover_by_cause"][name], rel_tol=1e-7, abs_tol=1e-8)
           for name, value in cause_totals.items()):
        raise ValueError("V3 action-switch turnover does not match registry")
    return {
        "steps": len(frame), "execution_dates": starts.astype(str).tolist(),
        "valuation_dates": ends.astype(str).tolist(),
        "start_date": str(starts.iloc[0].date()), "end_date": str(ends.iloc[-1].date()),
        "annualized_mean_gross_excess_return": float(np.mean(gross - cash) * 252),
        "annualized_mean_net_excess_return": float(np.mean(net - cash) * 252),
        "annualized_mean_cost_drag": float(np.mean(cost / nav) * 252),
        "turnover_by_cause": cause_totals,
    }


def diagnose(root: Path, data_root: Path, stage: str) -> dict:
    raw = load_contract(root)
    records = _records(root)
    spec = raw[stage]
    expected = _expected_ids(raw, stage, records)
    stage_records = validate_stage_completeness(
        records, stage, expected_experiment_ids=expected, expected_folds=FOLDS,
        expected_seeds=spec["seeds"], expected_timesteps=spec["timesteps_per_fold_seed"],
    )
    control_id = CONTROL if stage == "screening" else f"v3_confirmation_{CONTROL}"
    by_id = {item["experiment_id"]: item for item in stage_records}
    control = by_id[control_id]
    market_path = data_root / "data/processed/market_base_with_splits.csv"
    events_path = data_root / "data/interim/distribution_events.csv"
    if control["data_sha256"] != {
        "processed_market": sha256_file(market_path),
        "distribution_events": sha256_file(events_path),
    }:
        raise ValueError("V3 data files do not match frozen training inputs")
    if control["split_sha256"] != sha256_file(root / "configs/v2_rolling_folds.yaml"):
        raise ValueError("V3 folds do not match frozen training inputs")
    market, events = _development_only(pd.read_csv(market_path), pd.read_csv(events_path), "2022-01-01")
    folds = load_rolling_folds(root / "configs/v2_rolling_folds.yaml")
    if audit_folds(folds, market, events, ticker="SPY").status != "PASS":
        raise ValueError("V3 diagnostic fold audit failed")
    baselines = {}
    for fold in folds.folds:
        bundle = prepare_asset_bundle(
            market, events, folds.split_contract_for(fold), ticker="SPY", partition="validation"
        )
        baselines[fold.fold_id] = {
            policy: run_baseline(bundle, policy, transaction_cost_bps=10.0)
            for policy in ("cash", "buy_and_hold", "constant_50")
        }
    audited = {}
    for record in stage_records:
        for row in record["validation_metrics"]["per_fold_seed"]:
            key = (record["experiment_id"], row["fold"], row["seed"])
            value = audit_trajectory(root, row)
            if abs(value["annualized_mean_net_excess_return"] - row["annualized_mean_excess_return"]) > 1e-8:
                raise ValueError(f"V3 net excess does not match registry: {key}")
            base = baselines[row["fold"]]["cash"]
            if (value["steps"] != base["steps"] or value["start_date"] != base["start_date"]
                    or value["end_date"] != base["end_date"]):
                raise ValueError(f"V3 baseline timeline mismatch: {key}")
            audited[key] = value
    pairs = []
    control_rows = {(r["fold"], r["seed"]): r for r in control["validation_metrics"]["per_fold_seed"]}
    for record in stage_records:
        if record["experiment_id"] == control_id:
            continue
        for row in record["validation_metrics"]["per_fold_seed"]:
            key = (row["fold"], row["seed"])
            parent = control_rows[key]
            c = audited[(record["experiment_id"], *key)]
            p = audited[(control_id, *key)]
            if c["execution_dates"] != p["execution_dates"] or c["valuation_dates"] != p["valuation_dates"]:
                raise ValueError(f"V3 paired dates differ: {record['experiment_id']}, {key}")
            pairs.append({
                "candidate": record["experiment_id"], "fold": key[0], "seed": key[1],
                "eligible_candidate": row["eligible"], "eligible_control": parent["eligible"],
                "delta_sharpe_excess_cash": None if row["sharpe_excess_cash"] is None or parent["sharpe_excess_cash"] is None
                    else row["sharpe_excess_cash"] - parent["sharpe_excess_cash"],
                "delta_annualized_net_excess": c["annualized_mean_net_excess_return"] - p["annualized_mean_net_excess_return"],
                "delta_annualized_gross_excess": c["annualized_mean_gross_excess_return"] - p["annualized_mean_gross_excess_return"],
                "delta_annualized_cost_drag": c["annualized_mean_cost_drag"] - p["annualized_mean_cost_drag"],
                "delta_switch_turnover": c["turnover_by_cause"]["action_change"] - p["turnover_by_cause"]["action_change"],
                "delta_total_turnover": row["total_turnover"] - parent["total_turnover"],
            })
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage, "known_period_accessed": False,
        "purpose": "read_only_paired_diagnostic_not_selector",
        "v3_control": control_id, "records": expected,
        "same_fold_baselines": baselines, "paired_candidate_minus_control": pairs,
    }
