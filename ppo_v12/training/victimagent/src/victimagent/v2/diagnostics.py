"""Read-only, post-hoc V2 validation diagnostics; never selects a model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GROUPS = ("initial_entry", "action_change", "same_action")
REQUIRED = (
    "executed_action", "execution_date", "valuation_date", "turnover",
    "trade_notional", "transaction_cost", "start_nav", "gross_return",
    "net_return", "cash_benchmark_return",
)


def read_trajectory(root: Path, relative_path: str) -> pd.DataFrame:
    """Load only a registry-referenced V2 validation CSV inside this workspace."""
    root = root.resolve()
    allowed = (root / "artifacts" / "trajectories" / "v2").resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(allowed) or path.suffix != ".csv" or "_validation" not in path.stem:
        raise ValueError(f"not a V2 validation trajectory: {relative_path}")
    return pd.read_csv(path)


def decompose_turnover(frame: pd.DataFrame, *, expected_turnover: float | None = None,
                       expected_cost: float | None = None) -> dict[str, Any]:
    """Initial entry is separate: there is no prior action to change from."""
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"trajectory missing {sorted(missing)}")
    if frame.empty:
        raise ValueError("empty validation trajectory")
    numeric = frame.loc[:, ["turnover", "trade_notional", "transaction_cost", "start_nav",
                             "gross_return", "net_return", "cash_benchmark_return"]].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite ledger value")
    turnover, notional, cost, nav, gross, net, cash = numeric.T
    if (turnover < 0).any() or (cost < 0).any() or (nav <= 0).any():
        raise ValueError("invalid turnover, cost or starting NAV")
    if not np.allclose(turnover, np.abs(notional) / nav, rtol=1e-7, atol=1e-9):
        raise ValueError("turnover does not reconcile with trade notional / starting NAV")
    if not np.allclose(gross - net, cost / nav, rtol=1e-7, atol=1e-9):
        raise ValueError("gross-to-net return does not reconcile with cost / starting NAV")
    actions = pd.to_numeric(frame["executed_action"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(actions).all() or not np.array_equal(actions, actions.astype(int)):
        raise ValueError("invalid executed actions")
    dates = pd.to_datetime(frame["execution_date"], errors="raise")
    valuations = pd.to_datetime(frame["valuation_date"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing or (
        valuations < dates
    ).any():
        raise ValueError("invalid execution/valuation timeline")
    labels = np.full(len(frame), "same_action", dtype=object)
    labels[0] = "initial_entry"
    labels[1:] = np.where(actions[1:] != actions[:-1], "action_change", "same_action")
    groups = {}
    for label in GROUPS:
        mask = labels == label
        trade_mask = mask & (np.abs(notional) > 1e-8)
        groups[label] = {
            "decisions": int(mask.sum()),
            "trades": int(trade_mask.sum()),
            "turnover": float(turnover[mask].sum()),
            "transaction_cost": float(cost[mask].sum()),
            "median_trade_notional_abs": float(np.median(np.abs(notional[trade_mask])))
            if trade_mask.any() else 0.0,
        }
    total = float(turnover.sum())
    total_cost = float(cost.sum())
    for actual, expected, name in (
        (total, expected_turnover, "turnover"),
        (total_cost, expected_cost, "cost"),
    ):
        if expected is not None and not math.isclose(actual, float(expected), rel_tol=1e-7, abs_tol=1e-6):
            raise ValueError(
                f"{name} does not match registry metrics: "
                f"trajectory={actual:.12g}, registry={float(expected):.12g}"
            )
    return {
        "steps": len(frame),
        "start_execution_date": str(dates.iloc[0].date()),
        "end_valuation_date": str(valuations.iloc[-1].date()),
        "total_turnover": total,
        "total_transaction_cost": total_cost,
        "annualized_mean_gross_excess_return": float(np.mean(gross - cash) * 252.0),
        "annualized_mean_net_excess_return": float(np.mean(net - cash) * 252.0),
        "annualized_mean_cost_drag": float(np.mean(cost / nav) * 252.0),
        "groups": groups,
        "action_change_fraction_of_total_turnover": groups["action_change"]["turnover"] / total if total else None,
        "same_action_fraction_of_total_turnover": groups["same_action"]["turnover"] / total if total else None,
    }


def rows_by_fold_seed(record: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows = record["validation_metrics"]["per_fold_seed"]
    indexed = {(str(r["fold"]), int(r["seed"])): r for r in rows}
    expected = {(str(f), int(s)) for f in record["folds"] for s in record["seeds"]}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError(f"incomplete or duplicate fold-seed rows: {record['experiment_id']}")
    return indexed


def paired_differences(candidate: dict[str, Any], parent: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """All rows are retained; unlike Gate E0, do not filter on the outcome."""
    if candidate["experiment_id"] == parent["experiment_id"]:
        raise ValueError("candidate equals parent")
    keys = ("stage", "git_commit", "config_sha256", "split_sha256", "data_sha256", "training_budget")
    differences = [key for key in keys if candidate[key] != parent[key]]
    if differences:
        raise ValueError(f"cannot pair mixed provenance/budget: {differences}")
    if candidate.get("known_period_accessed") is not False or parent.get("known_period_accessed") is not False:
        raise ValueError("known-period record cannot enter validation diagnostics")
    child_rows, parent_rows = rows_by_fold_seed(candidate), rows_by_fold_seed(parent)
    if set(child_rows) != set(parent_rows):
        raise ValueError("candidate/parent fold-seed sets differ")
    paired = []
    for fold, seed in sorted(child_rows):
        c, p = child_rows[fold, seed], parent_rows[fold, seed]
        c_frame = read_trajectory(root, c["trajectory_path"])
        p_frame = read_trajectory(root, p["trajectory_path"])
        c_audit = decompose_turnover(c_frame, expected_turnover=c["total_turnover"])
        p_audit = decompose_turnover(p_frame, expected_turnover=p["total_turnover"])
        for date_col in ("execution_date", "valuation_date"):
            if c_frame[date_col].tolist() != p_frame[date_col].tolist():
                raise ValueError(f"validation timeline differs: {fold}, seed={seed}")
        row: dict[str, Any] = {"fold": fold, "seed": seed,
                               "eligible_candidate": c["eligible"], "eligible_parent": p["eligible"]}
        row["delta_annualized_gross_excess_return"] = (
            c_audit["annualized_mean_gross_excess_return"] - p_audit["annualized_mean_gross_excess_return"]
        )
        row["delta_annualized_cost_drag"] = (
            c_audit["annualized_mean_cost_drag"] - p_audit["annualized_mean_cost_drag"]
        )
        for metric, output in (
            ("sharpe_excess_cash", "delta_sharpe_excess_cash"),
            ("annualized_mean_excess_return", "delta_annualized_excess_return"),
            ("total_turnover", "delta_total_turnover"),
        ):
            c_value, p_value = c[metric], p[metric]
            row[output] = None if c_value is None or p_value is None else float(c_value) - float(p_value)
        paired.append(row)
    return paired


def summarize_pairs(paired: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive medians only; no Gate E0 acceptance or winner computation."""
    if not paired:
        raise ValueError("no paired fold-seed results")
    fields = ("delta_sharpe_excess_cash", "delta_annualized_excess_return", "delta_total_turnover",
              "delta_annualized_gross_excess_return", "delta_annualized_cost_drag")
    return {
        field: {"median": float(np.median(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)),
                "positive_count": int(sum(v > 0 for v in vals)), "count": len(vals)}
        if (vals := [row[field] for row in paired if row[field] is not None]) else None
        for field in fields
    }
