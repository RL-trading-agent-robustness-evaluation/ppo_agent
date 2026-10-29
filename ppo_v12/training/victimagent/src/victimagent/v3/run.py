"""V3 pre-2022 paired DQN study; never modifies a V2 artifact or registry."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import yaml

from victimagent.data.real_adapter import prepare_asset_bundle
from victimagent.dqn import evaluate_deterministic, parameter_sha256, seed_everything
from victimagent.environment import SingleAssetTradingEnvV1
from victimagent.evaluation import optimizer_sha256, performance_metrics, scaler_sha256
from victimagent.v2.folds import EPISODE_DECISIONS, audit_folds, load_rolling_folds
from victimagent.v2.registry import sha256_file
from victimagent.v2.run import _development_only, _responsiveness_diagnostics, build_v2_dqn
from victimagent.v2.select import select_stage_winner
from victimagent.v3.execution import TradeDecisionEnv

FOLDS = ("fold_1", "fold_2", "fold_3")
CONTROL = "v3_control_daily_target"


def load_contract(root: Path) -> dict[str, Any]:
    raw = yaml.safe_load((root / "configs/v3_trade_decision.yaml").read_text(encoding="utf-8"))
    if raw["status"] != "preregistered_before_v3_training":
        raise ValueError("V3 config is not preregistered")
    if raw["runner_status"] != "toy_gate_passed_local_fold_audit_required":
        raise ValueError("V3 toy audit is not yet complete")
    names = [r["id"] for r in raw["candidates"]]
    if len(names) != len(set(names)) or CONTROL not in names:
        raise ValueError("invalid declared V3 candidate IDs")
    if tuple(raw["screening"]["folds"]) != FOLDS or tuple(raw["confirmation"]["folds"]) != FOLDS:
        raise ValueError("V3 fold contract changed")
    if str(raw["immutable"]["selector_period_end"]) != "2021-12-31":
        raise ValueError("V3 forbidden period changed")
    if raw["immutable"]["primary_cost_bps"] != 10 or raw["immutable"]["exposure_action_weights"] != [0, 0.5, 1]:
        raise ValueError("V3 cost or action contract changed")
    return raw


def _research_clean(root: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", "src", "scripts", "configs", "pyproject.toml"],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("V3 research code/config must be committed before training")


def _frozen_v2_parent(root: Path, raw: dict) -> dict[str, str]:
    expected = {
        "action_space": "v2_action_space_ternary_0_05_1",
        "observation_ablation": "v2_observation_ablation_observation_v1_full",
        "finalist_confirmation": raw["parent_result"],
    }
    fingerprints = {}
    for stage, winner in expected.items():
        path = root / "reports/v2" / f"selection_{stage}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["winner_experiment_id"] != winner or report["known_period_accessed"] is not False:
            raise ValueError(f"V3 frozen V2 parent mismatch: {stage}")
        fingerprints[stage] = sha256_file(path)
    return fingerprints


def _sha(root: Path, data_root: Path) -> dict[str, Any]:
    return {
        "config_path": "configs/v3_trade_decision.yaml",
        "config_sha256": sha256_file(root / "configs/v3_trade_decision.yaml"),
        "split_sha256": sha256_file(root / "configs/v2_rolling_folds.yaml"),
        "v2_parent_hyperparameter_config_sha256": sha256_file(root / "configs/dqn_v2_search.yaml"),
        "data_sha256": {
            "processed_market": sha256_file(data_root / "data/processed/market_base_with_splits.csv"),
            "distribution_events": sha256_file(data_root / "data/interim/distribution_events.csv"),
        },
    }


def _commit(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


def _records(root: Path) -> list[dict[str, Any]]:
    path = root / "experiments/v3_registry.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _expected_ids(raw: dict, stage: str, records: list[dict]) -> list[str]:
    if stage == "screening":
        return [arm["id"] for arm in raw["candidates"]]
    winner = select_v3(raw, records, "screening")["winner_experiment_id"]
    return [f"v3_confirmation_{name}" for name in dict.fromkeys((CONTROL, winner))]


def select_v3(raw: dict, records: list[dict], stage: str) -> dict[str, Any]:
    if stage not in ("screening", "confirmation"):
        raise ValueError("unknown V3 selection stage")
    spec = raw[stage]
    ids = _expected_ids(raw, stage, records) if stage == "confirmation" else [
        item["id"] for item in raw["candidates"]
    ]
    result = select_stage_winner(
        records, stage, expected_experiment_ids=ids, expected_folds=FOLDS,
        expected_seeds=spec["seeds"], expected_timesteps=spec["timesteps_per_fold_seed"],
    ).to_dict()
    result["known_period_accessed"] = False
    result["expected_ids"] = ids
    return result


def reserve_attempt(root: Path, experiment_id: str, provenance: dict[str, Any]) -> Path:
    """O_EXCL prevents duplicates or overwrites even after a crashed attempt."""
    directory = root / "experiments/v3_attempts"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f"{experiment_id}.json"
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"status": "reserved", "experiment_id": experiment_id,
                       "created_at_utc": datetime.now(timezone.utc).isoformat(), **provenance}, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Marker is intentionally retained on interrupted or failed writes.
        raise
    return marker


def _append_record(root: Path, record: dict[str, Any]) -> None:
    path = root / "experiments/v3_registry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _environment(bundle, rule: str, *, training: bool):
    base = SingleAssetTradingEnvV1(
        bundle.market_true, bundle.market_obs, bundle.observation_builder,
        initial_cash=1_000_000.0, transaction_cost_bps=10.0,
        episode_length=EPISODE_DECISIONS if training else None,
        random_start=training, dividend_events=bundle.dividend_events,
        split_events=bundle.split_events, action_weights=(0.0, 0.5, 1.0),
    )
    return TradeDecisionEnv(base, rule)


def _callback():
    from stable_baselines3.common.callbacks import BaseCallback

    class FiniteLossCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.losses = []

        def _on_step(self) -> bool:
            loss = self.model.logger.name_to_value.get("train/loss")
            if loss is not None:
                if not np.isfinite(float(loss)):
                    raise FloatingPointError("V3 training produced a non-finite loss")
                self.losses.append(float(loss))
            return True

    return FiniteLossCallback()


def _train_one(*, root: Path, rule: str, hp: dict, fold: str, seed: int,
               timesteps: int, train_bundle, val_bundle, experiment_id: str) -> dict:
    from stable_baselines3 import DQN

    checkpoint = root / f"artifacts/checkpoints/v3/{experiment_id}_{fold}_seed{seed}.zip"
    trajectory = root / f"artifacts/trajectories/v3/{experiment_id}_{fold}_seed{seed}_validation.csv"
    if checkpoint.exists() or trajectory.exists():
        raise FileExistsError(f"V3 artifact already exists: {experiment_id}/{fold}/seed{seed}")
    seed_everything(seed)
    model = build_v2_dqn(_environment(train_bundle, rule, training=True), hp, seed=seed)
    pre_model = parameter_sha256(model)
    pre_scaler = scaler_sha256(train_bundle.observation_builder)
    callback = _callback()
    model.learn(total_timesteps=timesteps, callback=callback)
    if not callback.losses or not np.isfinite(callback.losses).all():
        raise RuntimeError("V3 training did not produce finite logged losses")
    if parameter_sha256(model) == pre_model:
        raise RuntimeError("V3 training did not update the policy")
    if scaler_sha256(train_bundle.observation_builder) != pre_scaler:
        raise RuntimeError("V3 training changed train-only scaler")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    env = _environment(val_bundle, rule, training=False)
    trained = DQN.load(checkpoint, env=env, device="cpu")
    before = (parameter_sha256(trained), optimizer_sha256(trained),
              int(trained.replay_buffer.size()), scaler_sha256(val_bundle.observation_builder))
    audit = evaluate_deterministic(trained, env, seed=seed)
    responsive = _responsiveness_diagnostics(
        trained, env, seed=seed, context_width=len(val_bundle.observation_builder.context_features)
    )
    after = (parameter_sha256(trained), optimizer_sha256(trained),
             int(trained.replay_buffer.size()), scaler_sha256(val_bundle.observation_builder))
    rows = audit.pop("trajectory")
    if not rows or any(row.get("trade_decision_rule") != rule for row in rows):
        raise AssertionError("V3 trajectory lacks the selected rule")
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(trajectory, index=False)
    metrics = performance_metrics(rows)
    gross = np.asarray([float(item["gross_return"]) for item in rows])
    net = np.asarray([float(item["net_return"]) for item in rows])
    cash = np.asarray([float(item["cash_benchmark_return"]) for item in rows])
    metrics["annualized_mean_gross_excess_return"] = float(np.mean(gross - cash) * 252)
    metrics["annualized_mean_cost_drag"] = float(np.mean(gross - net) * 252)
    previous_executed = None
    turnover_by_cause = {"initial_entry": 0.0, "action_change": 0.0,
                         "same_action": 0.0, "skipped_order": 0.0}
    for item in rows:
        if item["order_placed"]:
            current = int(item["executed_action"])
            cause = ("initial_entry" if previous_executed is None else
                     "action_change" if current != previous_executed else "same_action")
            previous_executed = current
        else:
            cause = "skipped_order"
            if float(item["turnover"]) != 0 or float(item["transaction_cost"]) != 0:
                raise AssertionError("skipped order generated turnover or transaction cost")
        turnover_by_cause[cause] += float(item["turnover"])
    if not np.isclose(sum(turnover_by_cause.values()), metrics["total_turnover"], atol=1e-9):
        raise AssertionError("V3 turnover decomposition does not reconcile")
    metrics["turnover_by_cause"] = turnover_by_cause
    metrics["skipped_order_count"] = sum(not item["order_placed"] for item in rows)
    gates = {
        "finite_training_loss": bool(callback.losses) and bool(np.isfinite(callback.losses).all()),
        "reward_nav_invariant": abs(float(audit["reward_nav_invariant_error"])) <= 1e-10,
        "natural_termination": bool(audit["terminated"]) and not bool(audit["truncated"]),
        "nav_finite_positive": bool(audit["nav_finite_positive"]),
        "q_values_finite": bool(audit["q_values_finite"]),
        "evaluation_state_unchanged": before == after,
        "state_responsive": bool(responsive["eligible"]),
    }
    return {
        "fold": fold, "seed": seed, "eligible": all(gates.values()),
        "technical_gates": gates, "responsiveness": responsive,
        "checkpoint_path": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "trajectory_path": trajectory.relative_to(root).as_posix(),
        "trajectory_sha256": sha256_file(trajectory),
        **metrics,
    }


def run_v3(root: Path, data_root: Path, stage: str, candidate_id: str) -> dict[str, Any]:
    """Execute one declared arm, reserving an attempt before any model artifact."""
    raw = load_contract(root)
    _research_clean(root)
    parent_report_sha256 = _frozen_v2_parent(root, raw)
    if stage not in ("screening", "confirmation"):
        raise ValueError("stage must be screening or confirmation")
    records = _records(root)
    permitted = _expected_ids(raw, stage, records)
    experiment_id = candidate_id if stage == "screening" else f"v3_confirmation_{candidate_id}"
    if experiment_id not in permitted:
        raise ValueError(f"{experiment_id} is not a declared {stage} arm: {permitted}")
    if any(item.get("experiment_id") == experiment_id for item in records):
        raise FileExistsError(f"V3 experiment already recorded: {experiment_id}")
    if (root / f"experiments/v3_attempts/{experiment_id}.json").exists():
        raise FileExistsError(f"V3 attempt already reserved: {experiment_id}")
    if (root / "artifacts/checkpoints/v3").exists() and list(
        (root / "artifacts/checkpoints/v3").glob(f"{experiment_id}_*.zip")
    ):
        raise FileExistsError("V3 orphan checkpoint requires audit before retry")
    if (root / "artifacts/trajectories/v3").exists() and list(
        (root / "artifacts/trajectories/v3").glob(f"{experiment_id}_*.csv")
    ):
        raise FileExistsError("V3 orphan trajectory requires audit before retry")

    hashes = _sha(root, data_root)
    git_sha = _commit(root)
    spec = raw[stage]
    market_path = data_root / "data/processed/market_base_with_splits.csv"
    event_path = data_root / "data/interim/distribution_events.csv"
    market = pd.read_csv(market_path)
    events = pd.read_csv(event_path)
    market, events = _development_only(market, events, "2022-01-01")
    folds = load_rolling_folds(root / "configs/v2_rolling_folds.yaml")
    audit = audit_folds(folds, market, events, ticker="SPY")
    if audit.status != "PASS":
        raise RuntimeError("V3 fold audit failed; no training attempted")
    if tuple(f.fold_id for f in folds.folds) != FOLDS:
        raise ValueError("V3 fold order changed")

    reservation = reserve_attempt(root, experiment_id, {
        "git_commit": git_sha, "stage": stage, "known_period_accessed": False,
        "v2_parent_report_sha256": parent_report_sha256, **hashes,
    })
    rows: list[dict] = []
    try:
        hp_stage = next(s for s in yaml.safe_load(
            (root / "configs/dqn_v2_search.yaml").read_text(encoding="utf-8")
        )["stages"] if s["id"] == "dqn_hyperparameters")
        hp = {**hp_stage["fixed"], **hp_stage["baseline"]}
        for fold in folds.folds:
            contract = folds.split_contract_for(fold)
            train_bundle = prepare_asset_bundle(market, events, contract, ticker="SPY", partition="train")
            val_bundle = prepare_asset_bundle(market, events, contract, ticker="SPY", partition="validation")
            for seed in spec["seeds"]:
                rows.append(_train_one(root=root, rule=candidate_id, hp=hp,
                                       fold=fold.fold_id, seed=int(seed),
                                       timesteps=int(spec["timesteps_per_fold_seed"]),
                                       train_bundle=train_bundle, val_bundle=val_bundle,
                                       experiment_id=experiment_id))
        status = "pass" if all(r["eligible"] for r in rows) else "fail"
        error = None
    except BaseException as exc:
        status, error = "fail", f"{type(exc).__name__}: {exc}"
        # Keep the reservation and partial artifact paths for incident analysis.
        raise
    finally:
        record = {
            "experiment_id": experiment_id, "stage": stage, "candidate_id": candidate_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status, "error": error, "known_period_accessed": False,
            "git_commit": git_sha, **hashes,
            "v2_parent_report_sha256": parent_report_sha256,
            "folds": list(FOLDS), "seeds": list(spec["seeds"]),
            "training_budget": {"timesteps_per_fold_seed": spec["timesteps_per_fold_seed"]},
            "validation_metrics": {
                "primary_metric": "median_sharpe_excess_lagged_cash",
                "per_fold_seed": rows,
            },
            "attempt_marker": reservation.relative_to(root).as_posix(),
        }
        _append_record(root, record)
    return record
