"""Read-only V3 in-sample versus rolling-validation checkpoint diagnostic.

This diagnostic is descriptive. It never trains, selects a checkpoint, or opens
the 2022–2025 known period. In-sample here means a deterministic full train
partition rollout of an already trained fold checkpoint, not the training log.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
import pandas as pd

from victimagent.baselines import run_baseline
from victimagent.data.real_adapter import prepare_asset_bundle
from victimagent.dqn import evaluate_deterministic, parameter_sha256
from victimagent.evaluation import optimizer_sha256, performance_metrics, scaler_sha256
from victimagent.v2.folds import audit_folds, load_rolling_folds
from victimagent.v2.registry import sha256_file
from victimagent.v2.run import _development_only
from victimagent.v2.select import validate_stage_completeness
from victimagent.v3.diagnostics import audit_trajectory
from victimagent.v3.run import FOLDS, _environment, _expected_ids, _records, load_contract


def summarize_trajectory(rows: list[dict]) -> dict:
    """Reconcile the ledger and return metrics comparable across partitions."""
    metrics = performance_metrics(rows)
    gross = np.asarray([float(row["gross_return"]) for row in rows])
    net = np.asarray([float(row["net_return"]) for row in rows])
    cash = np.asarray([float(row["cash_benchmark_return"]) for row in rows])
    cost = np.asarray([float(row["transaction_cost"]) for row in rows])
    start_nav = np.asarray([float(row["start_nav"]) for row in rows])
    notional = np.asarray([float(row["trade_notional"]) for row in rows])
    turnover = np.asarray([float(row["turnover"]) for row in rows])
    if not np.isfinite(np.stack((gross, net, cash, cost, start_nav, notional, turnover))).all():
        raise ValueError("non-finite evaluation ledger")
    if (start_nav <= 0).any() or not np.allclose(turnover, np.abs(notional) / start_nav, atol=1e-9, rtol=1e-7):
        raise ValueError("evaluation turnover does not reconcile")
    if not np.allclose(gross - net, cost / start_nav, atol=1e-9, rtol=1e-7):
        raise ValueError("evaluation gross/net cost does not reconcile")
    causes = {key: 0.0 for key in ("initial_entry", "action_change", "same_action", "skipped_order")}
    previous = None
    for row in rows:
        if row["order_placed"]:
            action = int(row["executed_action"])
            cause = "initial_entry" if previous is None else "action_change" if action != previous else "same_action"
            previous = action
        else:
            cause = "skipped_order"
            if abs(float(row["turnover"])) > 1e-12 or abs(float(row["transaction_cost"])) > 1e-9:
                raise ValueError("skipped V3 order incurred a cost")
        causes[cause] += float(row["turnover"])
    if not math.isclose(sum(causes.values()), metrics["total_turnover"], abs_tol=1e-8):
        raise ValueError("turnover causes do not sum to total turnover")
    metrics.update({
        "annualized_mean_gross_excess_return": float(np.mean(gross - cash) * 252),
        "annualized_mean_cost_drag": float(np.mean(cost / start_nav) * 252),
        "mean_daily_turnover": float(np.mean(turnover)),
        "turnover_by_cause": causes,
    })
    if not math.isclose(
        metrics["annualized_mean_gross_excess_return"] - metrics["annualized_mean_cost_drag"],
        metrics["annualized_mean_excess_return"], abs_tol=1e-8,
    ):
        raise ValueError("annualized gross minus cost is not net excess")
    return metrics


def _checked_checkpoint(root: Path, row: dict) -> Path:
    base = (root / "artifacts/checkpoints/v3").resolve()
    path = (root / row["checkpoint_path"]).resolve()
    if not path.is_relative_to(base) or path.suffix != ".zip":
        raise ValueError("checkpoint must be an existing V3 artifact")
    if sha256_file(path) != row["checkpoint_sha256"]:
        raise ValueError(f"checkpoint SHA-256 mismatch: {path.name}")
    return path


def _rollout(checkpoint: Path, bundle, rule: str, seed: int) -> dict:
    from stable_baselines3 import DQN

    env = _environment(bundle, rule, training=False)
    model = DQN.load(checkpoint, env=env, device="cpu")
    before = (parameter_sha256(model), optimizer_sha256(model),
              int(model.replay_buffer.size()), int(model.replay_buffer.pos),
              scaler_sha256(bundle.observation_builder))
    audit = evaluate_deterministic(model, env, seed=seed)
    after = (parameter_sha256(model), optimizer_sha256(model),
             int(model.replay_buffer.size()), int(model.replay_buffer.pos),
             scaler_sha256(bundle.observation_builder))
    if before != after or not audit["parameters_unchanged"] or not audit["replay_unchanged"]:
        raise ValueError("evaluation modified model, optimizer, buffer, or scaler")
    if not audit["terminated"] or audit["truncated"] or not audit["q_values_finite"] or not audit["nav_finite_positive"]:
        raise ValueError("evaluation did not terminate naturally with finite values")
    if abs(audit["reward_nav_invariant_error"]) > 1e-10:
        raise ValueError("reward–NAV invariant failed")
    rows = audit["trajectory"]
    if not rows or any(row.get("trade_decision_rule") != rule for row in rows):
        raise ValueError("trajectory does not use frozen V3 execution rule")
    return summarize_trajectory(rows)


def _same_window(metrics: dict, baseline: dict) -> None:
    if any(metrics[k] != baseline[k] for k in ("steps", "start_date", "end_date")):
        raise ValueError("policy and same-fold baseline timelines differ")


def paired_gap(train: dict, validation: dict, train_baselines: dict, val_baselines: dict) -> dict:
    """Use period-normalized differences; raw wealth is never compared across lengths."""
    for metrics, baselines in ((train, train_baselines), (validation, val_baselines)):
        for baseline in baselines.values():
            _same_window(metrics, baseline)
    result = {}
    for key in ("sharpe_excess_cash", "annualized_mean_excess_return",
                "annualized_mean_gross_excess_return", "annualized_mean_cost_drag",
                "mean_daily_turnover", "average_exposure"):
        left, right = train[key], validation[key]
        result[f"train_minus_validation_{key}"] = None if left is None or right is None else left - right
    for partition, metrics, baselines in (("train", train, train_baselines),
                                          ("validation", validation, val_baselines)):
        for name in ("cash", "buy_and_hold", "constant_50"):
            baseline = baselines[name]
            for key in ("annualized_mean_excess_return", "sharpe_excess_cash"):
                a, b = metrics[key], baseline[key]
                result[f"{partition}_minus_{name}_{key}"] = None if a is None or b is None else a - b
    for name in ("buy_and_hold", "constant_50"):
        for key in ("annualized_mean_excess_return", "sharpe_excess_cash"):
            t = result[f"train_minus_{name}_{key}"]
            v = result[f"validation_minus_{name}_{key}"]
            result[f"train_minus_validation_relative_{name}_{key}"] = None if t is None or v is None else t - v
    return result


def _median(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return float(np.median(values)) if values else None


def diagnose_train_validation(root: Path, data_root: Path) -> dict:
    raw, records = load_contract(root), _records(root)
    expected = _expected_ids(raw, "confirmation", records)
    stage_records = validate_stage_completeness(
        records, "confirmation", expected_experiment_ids=expected, expected_folds=FOLDS,
        expected_seeds=raw["confirmation"]["seeds"],
        expected_timesteps=raw["confirmation"]["timesteps_per_fold_seed"],
    )
    market_path = data_root / "data/processed/market_base_with_splits.csv"
    event_path = data_root / "data/interim/distribution_events.csv"
    data_sha = {"processed_market": sha256_file(market_path),
                "distribution_events": sha256_file(event_path)}
    config_sha = sha256_file(root / "configs/v3_trade_decision.yaml")
    split_sha = sha256_file(root / "configs/v2_rolling_folds.yaml")
    for record in stage_records:
        if record["known_period_accessed"] is not False or record["data_sha256"] != data_sha or record["config_sha256"] != config_sha or record["split_sha256"] != split_sha:
            raise ValueError("confirmation provenance differs from frozen development inputs")
    # Load only the pre-2022 development span before building any observation.
    market, events = _development_only(pd.read_csv(market_path), pd.read_csv(event_path), "2022-01-01")
    folds = load_rolling_folds(root / "configs/v2_rolling_folds.yaml")
    if tuple(f.fold_id for f in folds.folds) != FOLDS or audit_folds(folds, market, events, ticker="SPY").status != "PASS":
        raise ValueError("V3 rolling fold audit failed")
    baselines, bundles = {}, {}
    for fold in folds.folds:
        contract = folds.split_contract_for(fold)
        bundles[fold.fold_id] = {partition: prepare_asset_bundle(
            market, events, contract, ticker="SPY", partition=partition,
        ) for partition in ("train", "validation")}
        if scaler_sha256(bundles[fold.fold_id]["train"].observation_builder) != scaler_sha256(bundles[fold.fold_id]["validation"].observation_builder):
            raise ValueError("train and validation scaler fingerprints differ")
        baselines[fold.fold_id] = {partition: {policy: run_baseline(
            bundles[fold.fold_id][partition], policy, transaction_cost_bps=10.0,
        ) for policy in ("cash", "buy_and_hold", "constant_50")}
            for partition in ("train", "validation")}
    diagnostics = []
    for record in stage_records:
        for row in record["validation_metrics"]["per_fold_seed"]:
            fold, seed = row["fold"], int(row["seed"])
            expected_stem = f"{record['experiment_id']}_{fold}_seed{seed}"
            if (Path(row["checkpoint_path"]).name != f"{expected_stem}.zip"
                    or Path(row["trajectory_path"]).name != f"{expected_stem}_validation.csv"):
                raise ValueError("checkpoint or trajectory name does not match its fold, seed and experiment")
            checkpoint = _checked_checkpoint(root, row)
            original = audit_trajectory(root, row)  # validates immutable source trajectory
            pair = {partition: _rollout(checkpoint, bundles[fold][partition],
                                        record["candidate_id"], seed)
                    for partition in ("train", "validation")}
            replay = pair["validation"]
            for key in ("annualized_mean_excess_return", "total_turnover", "total_transaction_cost"):
                if not math.isclose(replay[key], row[key], abs_tol=1e-6, rel_tol=1e-7):
                    raise ValueError(f"validation replay differs from original registry: {record['experiment_id']}/{fold}/{seed}/{key}")
            if not math.isclose(replay["annualized_mean_gross_excess_return"], original["annualized_mean_gross_excess_return"], abs_tol=1e-8):
                raise ValueError("validation replay differs from hashed original trajectory")
            _same_window(replay, {"steps": original["steps"], "start_date": original["start_date"], "end_date": original["end_date"]})
            gap = paired_gap(pair["train"], replay, baselines[fold]["train"], baselines[fold]["validation"])
            diagnostics.append({"experiment_id": record["experiment_id"], "record_status": record["status"],
                                "fold": fold, "seed": seed, "original_eligible": row["eligible"],
                                "checkpoint_sha256": row["checkpoint_sha256"],
                                "original_validation_trajectory_sha256": row["trajectory_sha256"],
                                "train": pair["train"], "validation": replay, "gap": gap})
    aggregates = {}
    keys = ("train_minus_validation_sharpe_excess_cash",
            "train_minus_validation_annualized_mean_excess_return",
            "train_minus_validation_annualized_mean_gross_excess_return",
            "train_minus_validation_annualized_mean_cost_drag",
            "train_minus_validation_relative_buy_and_hold_annualized_mean_excess_return",
            "train_minus_validation_relative_constant_50_annualized_mean_excess_return")
    for experiment in expected:
        subset = [d for d in diagnostics if d["experiment_id"] == experiment]
        aggregates[experiment] = {"all_15_included": len(subset) == 15,
                                  "original_ineligible": sum(not d["original_eligible"] for d in subset),
                                  "median_gap": {key: _median([d["gap"] for d in subset], key) for key in keys},
                                  "per_fold": {fold: {key: _median([d["gap"] for d in subset if d["fold"] == fold], key)
                                                       for key in keys} for fold in FOLDS}}
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "read_only_in_sample_diagnostic_not_model_selection",
            "known_period_accessed": False, "partitions": ["train", "validation"],
            "input_sha256": {"v3_registry": sha256_file(root / "experiments/v3_registry.jsonl"),
                             "v3_config": config_sha, "rolling_folds": split_sha, **data_sha},
            "original_training_commits": {record["experiment_id"]: record["git_commit"]
                                          for record in stage_records},
            "experiment_ids": expected, "same_period_baselines": baselines,
            "per_fold_seed": diagnostics, "summary": aggregates}
