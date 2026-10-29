from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd

from victimagent.evaluation import optimizer_sha256, scaler_fingerprint, scaler_sha256
from victimagent.v2.folds import load_rolling_folds
from victimagent.v2.registry import sha256_file
from victimagent.v4.evaluation import rollout
from victimagent.v8 import study as v8
from victimagent.v9 import study as v9

from .config import (ARMS, EXPECTED_ACTUAL_TIMESTEPS, EXPECTED_PIN, FOLDS,
                     KNOWN_BUY_AND_HOLD, RAW_FILES, SEEDS, PPOConfigV9,
                     missing_raw_files)

RECORDS = Path("experiments/v9_ppo/records")
ATTEMPTS = Path("experiments/v9_ppo/attempts")
REPORTS = Path("reports/v9_ppo")
FINAL_RECORDS = Path("experiments/v9_ppo/final_records")
FINAL_ATTEMPTS = Path("experiments/v9_ppo/final_attempts")
EXPECTED_BUY_AND_HOLD = {
    "fold_1": 1.1366146906263972,
    "fold_2": 1.1510616812248984,
    "fold_3": 1.9801822999404477,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, default=str)


def parameter_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def linear_schedule(initial: float, final: float):
    if not 0 < final < initial:
        raise ValueError("expected 0 < final < initial")

    def schedule(progress_remaining: float) -> float:
        progress = min(1.0, max(0.0, float(progress_remaining)))
        return final + progress * (initial - final)

    return schedule


def resolved_submodule_pin(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root / "victimagent"), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def git_commit(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def package_versions() -> dict[str, str]:
    return {name: version(dist) for name, dist in {
        "stable_baselines3": "stable-baselines3", "torch": "torch",
        "numpy": "numpy", "gymnasium": "gymnasium", "pandas": "pandas",
    }.items()}


def preflight(root: Path, config: PPOConfigV9, *, require_data: bool = True,
              require_clean: bool = False) -> dict[str, Any]:
    victim = root / "victimagent"
    resolved = resolved_submodule_pin(root)
    if resolved != EXPECTED_PIN:
        raise RuntimeError(f"victimagent pin {resolved} != required {EXPECTED_PIN}")
    missing = missing_raw_files(victim)
    if require_data and missing:
        raise FileNotFoundError("missing victimagent/data/raw files: " + ", ".join(missing))
    if require_clean:
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "src", "scripts",
             "configs", "pyproject.toml"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError("PPO V9 code/config must be committed:\n" + dirty)
        tracked_submodule_changes = subprocess.run(
            ["git", "-C", str(victim), "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if tracked_submodule_changes:
            raise RuntimeError("victimagent has tracked changes:\n" + tracked_submodule_changes)
    result: dict[str, Any] = {
        "status": "WAITING_FOR_DATA" if missing else "READY_FOR_DATA_BUILD",
        "victimagent_commit": resolved,
        "ppo_repo_commit": git_commit(root),
        "required_raw_files": list(RAW_FILES),
        "missing_raw_files": missing,
        "known_period_accessed": False,
        "ppo_config_sha256": sha256_file(config.path),
    }
    if not missing and require_data:
        upstream = v8.preflight(victim, victim, require_clean=False)
        expected = config.raw["shared_contract"]["expected_processed_sha256"]
        actual = upstream["data_build"]["processed_sha256"]
        market_rel = v8.contract(victim)["data"]["processed_market"]
        event_rel = v8.contract(victim)["data"]["processed_events"]
        if actual.get(market_rel) != expected["market"] or actual.get(event_rel) != expected["events"]:
            raise RuntimeError("V9 processed-data hashes do not match the preregistered contract")
        result.update(status="PASS", upstream=upstream)
    return result


def build_model(env, config: PPOConfigV9, *, seed: int, tensorboard_log: Path | None = None):
    import torch
    from stable_baselines3 import PPO

    hp = config.model
    activation = {"Tanh": torch.nn.Tanh, "ReLU": torch.nn.ReLU}[hp["activation_fn"]]
    return PPO(
        config.raw["policy"], env,
        learning_rate=linear_schedule(float(hp["learning_rate_initial"]),
                                      float(hp["learning_rate_final"])),
        n_steps=int(hp["n_steps"]), batch_size=int(hp["batch_size"]),
        n_epochs=int(hp["n_epochs"]), gamma=float(hp["gamma"]),
        gae_lambda=float(hp["gae_lambda"]), clip_range=float(hp["clip_range"]),
        normalize_advantage=bool(hp["normalize_advantage"]),
        ent_coef=float(hp["ent_coef"]), vf_coef=float(hp["vf_coef"]),
        max_grad_norm=float(hp["max_grad_norm"]),
        policy_kwargs={"net_arch": list(hp["policy_net_arch"]), "activation_fn": activation},
        seed=seed, device=config.raw["device"], verbose=1,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )


def _train(root: Path, config: PPOConfigV9, bundle, spy, arm: str, seed: int,
           requested_timesteps: int, label: str):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    class LossCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.losses: list[float] = []

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            value = self.model.logger.name_to_value.get("train/loss")
            if value is not None:
                self.losses.append(float(value))

    random.seed(seed)
    np.random.seed(seed)
    upstream_arm = config.upstream_arm(arm)
    env = v9._environment(bundle, spy, upstream_arm, training=True)
    if env.observation_space.shape != (20,) or env.action_space.n != 3:
        raise AssertionError("PPO did not receive the shared V9 20-D/3-action interface")
    checkpoint = root / "artifacts/checkpoints/v9_ppo" / f"{label}.zip"
    if checkpoint.exists():
        raise FileExistsError(f"checkpoint exists: {checkpoint}")
    tb = root / "tensorboard/v9_ppo" / label
    log = root / "artifacts/logs/v9_ppo" / label
    log.mkdir(parents=True, exist_ok=False)
    model = build_model(env, config, seed=seed, tensorboard_log=tb)
    model.set_logger(configure(str(log), ["csv", "tensorboard"]))
    before_param, before_scaler = parameter_sha256(model), scaler_sha256(bundle.observation_builder)
    callback, started = LossCallback(), time.time()
    model.learn(total_timesteps=requested_timesteps, callback=callback,
                reset_num_timesteps=True, progress_bar=False, tb_log_name="run")
    if not callback.losses or not np.isfinite(callback.losses).all():
        raise RuntimeError("PPO training did not produce finite losses")
    if parameter_sha256(model) == before_param:
        raise RuntimeError("PPO parameters did not update")
    if scaler_sha256(bundle.observation_builder) != before_scaler:
        raise RuntimeError("training mutated the fold-train-only scaler")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    return checkpoint, {
        "train_seconds": round(time.time() - started, 1),
        "requested_timesteps": requested_timesteps,
        "actual_timesteps": int(model.num_timesteps),
        "final_logged_loss": callback.losses[-1],
        "tensorboard_path": tb.relative_to(root).as_posix(),
        "training_log_path": log.relative_to(root).as_posix(),
    }


def responsiveness(model, observations: list[np.ndarray]) -> dict[str, Any]:
    matrix = np.asarray(observations, dtype=np.float32)
    original, _ = model.predict(matrix, deterministic=True)
    shifted = matrix.copy()
    shifted[:, :12] = np.roll(shifted[:, :12], 1, axis=0)
    changed, _ = model.predict(shifted, deterministic=True)
    ratio = float(np.mean(np.asarray(original) != np.asarray(changed)))
    return {
        "method": "cyclic_shift_first_12_market_and_rate_dimensions",
        "account_and_rule_dimensions_preserved": True,
        "action_change_ratio": ratio,
        "original_action_types": sorted(set(np.asarray(original).astype(int).tolist())),
        "perturbed_action_types": sorted(set(np.asarray(changed).astype(int).tolist())),
        "eligible": bool(ratio > 0.0),
    }


def _evaluate(root: Path, config: PPOConfigV9, checkpoint: Path, bundle, spy,
              arm: str, seed: int, label: str) -> dict[str, Any]:
    from stable_baselines3 import PPO

    upstream_arm = config.upstream_arm(arm)
    env = v9._environment(bundle, spy, upstream_arm, training=False)
    model = PPO.load(checkpoint, env=env, device="cpu")
    model.policy.set_training_mode(False)
    before = (parameter_sha256(model), optimizer_sha256(model),
              scaler_sha256(bundle.observation_builder))
    observations: list[np.ndarray] = []

    def choose(observation, _index):
        observations.append(np.asarray(observation).copy())
        action, _ = model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())

    rows, measures, error = rollout(env, choose, seed=seed)
    trace = v9._action_trace(rows, bundle, upstream_arm)
    after = (parameter_sha256(model), optimizer_sha256(model),
             scaler_sha256(bundle.observation_builder))
    response = responsiveness(model, observations)
    trajectory = root / "artifacts/trajectories/v9_ppo" / f"{label}_validation.csv"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    if trajectory.exists():
        raise FileExistsError(f"trajectory exists: {trajectory}")
    pd.DataFrame(rows).to_csv(trajectory, index=False)
    largest = max(trace["agent_action_distribution"].values())
    return {
        "technical_pass": bool(before == after and abs(error) <= 1e-10),
        "evaluation_state_unchanged": before == after,
        "reward_nav_error": error,
        "responsiveness": response,
        "state_responsive": response["eligible"],
        "near_constant_policy": bool(largest >= 0.95),
        "trajectory_path": trajectory.relative_to(root).as_posix(),
        "trajectory_sha256": sha256_file(trajectory),
        **trace, **measures,
    }


def run_cell(root: Path, config: PPOConfigV9, *, arm: str, fold_id: str, seed: int,
             requested_timesteps: int, tag: str = "", threads: int = 1) -> dict[str, Any]:
    if arm not in ARMS or fold_id not in FOLDS or seed not in SEEDS:
        raise ValueError("invalid PPO V9 arm, fold, or seed")
    formal = requested_timesteps == 500_000
    provenance = preflight(root, config, require_data=True, require_clean=formal)
    victim = root / "victimagent"
    v8._set_threads(threads)
    fold = next(f for f in load_rolling_folds(victim / "configs/v8_folds.yaml").folds
                if f.fold_id == fold_id)
    market, events = v8._v8_inputs(victim, victim)
    train, train_spy = v9.v9_bundle(market, events, fit=fold.train, select=fold.train)
    validation, val_spy = v9.v9_bundle(market, events, fit=fold.train, select=fold.validation)
    label = f"{arm}_{fold_id}_seed{seed}{tag}"
    attempt = root / ATTEMPTS / f"{label}.json"
    record_path = root / RECORDS / f"{label}.json"
    if attempt.exists() or record_path.exists():
        raise FileExistsError(f"PPO V9 cell already attempted: {label}")
    write_new(attempt, {"status": "reserved", "created_at_utc": now(), "arm": arm,
                        "fold": fold_id, "seed": seed, "formal": formal})
    result, status, error = {}, "FAIL", None
    checkpoint = None
    try:
        checkpoint, training = _train(root, config, train, train_spy, arm, seed,
                                      requested_timesteps, label)
        evaluation = _evaluate(root, config, checkpoint, validation, val_spy, arm, seed, label)
        baselines = v9._baselines_with_rule(validation, val_spy)
        bh = baselines["buy_and_hold"]["final_wealth"]
        if not math.isclose(bh, EXPECTED_BUY_AND_HOLD[fold_id], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"{fold_id} buy-and-hold {bh} != frozen DQN value")
        result = {**training, **evaluation, "baselines": baselines,
                  "log_relative_wealth_vs_buy_and_hold":
                      math.log(evaluation["final_wealth"] / bh)}
        status = "PASS" if evaluation["technical_pass"] else "FAIL"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_new(record_path, {
            "status": status, "error": error, "formal": formal,
            "algorithm": "PPO", "arm": config.upstream_arm(arm), "ppo_arm": arm,
            "fold": fold_id, "seed": seed, "total_timesteps": requested_timesteps,
            "created_at_utc": now(), "known_period_accessed": False,
            "preflight": provenance, "hyperparameters": config.model,
            "package_versions": package_versions(),
            "scaler": scaler_fingerprint(train.observation_builder),
            "train_window": train.metadata, "validation_window": validation.metadata,
            "checkpoint_path": checkpoint.relative_to(root).as_posix() if checkpoint else None,
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
            **result,
        })
    return json.loads(record_path.read_text(encoding="utf-8"))


def run_final_seed(root: Path, config: PPOConfigV9, *, seed: int,
                   threads: int = 1) -> dict[str, Any]:
    """Train one preregistered final seed without loading the known period."""
    if seed not in SEEDS:
        raise ValueError("invalid PPO V9 final seed")
    provenance = preflight(root, config, require_data=True, require_clean=True)
    victim = root / "victimagent"
    v8._set_threads(threads)
    market, events = v8._v8_inputs(victim, victim)  # defaults to before 2022
    bundle, spy = v9.v9_bundle(market, events, fit=v9.FINAL_WINDOW,
                               select=v9.FINAL_WINDOW)
    label = f"final_v9_gate_relative_reward_seed{seed}"
    attempt = root / FINAL_ATTEMPTS / f"{label}.json"
    record_path = root / FINAL_RECORDS / f"{label}.json"
    if attempt.exists() or record_path.exists():
        raise FileExistsError(f"PPO V9 final seed already attempted: {label}")
    write_new(attempt, {"status": "reserved", "created_at_utc": now(),
                        "seed": seed, "known_period_accessed": False})
    checkpoint, result, status, error = None, {}, "FAIL", None
    try:
        checkpoint, training = _train(
            root, config, bundle, spy, "ppo_v9_gate_relative", seed,
            int(config.raw["final"]["requested_timesteps_per_seed"]), label,
        )
        evaluation = _evaluate(root, config, checkpoint, bundle, spy,
                               "ppo_v9_gate_relative", seed, label)
        result = {**training, **evaluation}
        status = "PASS" if evaluation["technical_pass"] else "FAIL"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_new(record_path, {
            "status": status, "error": error, "algorithm": "PPO",
            "arm": "v9_gate_relative_reward", "seed": seed,
            "total_timesteps": int(config.raw["final"]["requested_timesteps_per_seed"]),
            "created_at_utc": now(), "known_period_accessed": False,
            "preflight": provenance, "hyperparameters": config.model,
            "package_versions": package_versions(),
            "scaler": scaler_fingerprint(bundle.observation_builder),
            "train_window": bundle.metadata,
            "checkpoint_path": checkpoint.relative_to(root).as_posix() if checkpoint else None,
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
            **result,
        })
    return json.loads(record_path.read_text(encoding="utf-8"))


def confirm_known_period(root: Path, config: PPOConfigV9) -> dict[str, Any]:
    """Perform the append-only, one-shot known-period confirmation for all seeds."""
    from stable_baselines3 import PPO

    preflight_result = preflight(root, config, require_data=True, require_clean=True)
    attempt = root / FINAL_ATTEMPTS / "known_period_confirmation.json"
    report = root / REPORTS / "KNOWN_PERIOD_CONFIRMATION.json"
    manifest_path = root / REPORTS / "FINAL_VICTIM_MANIFEST.json"
    if attempt.exists() or report.exists() or manifest_path.exists():
        raise FileExistsError("known-period confirmation has already been attempted")
    records = []
    for seed in SEEDS:
        path = root / FINAL_RECORDS / f"final_v9_gate_relative_reward_seed{seed}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("status") != "PASS" or row.get("known_period_accessed") is not False
                or row.get("actual_timesteps") != EXPECTED_ACTUAL_TIMESTEPS):
            raise RuntimeError(f"final seed is not eligible for confirmation: {path}")
        records.append(row)
    write_new(attempt, {"status": "reserved", "created_at_utc": now(),
                        "label": "known_period_confirmation_not_untouched_test"})

    victim = root / "victimagent"
    market, events = v8._v8_inputs(victim, victim, before=None)
    known, spy = v9.v9_bundle(market, events, fit=v9.FINAL_WINDOW, select=v8.KNOWN)
    baselines = v9._baselines_with_rule(known, spy)
    bh = baselines["buy_and_hold"]["final_wealth"]
    if not math.isclose(bh, KNOWN_BUY_AND_HOLD, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"known-period buy-and-hold {bh} != frozen DQN value")
    seed_results = []
    expected_scaler = scaler_sha256(known.observation_builder)
    expected_scaler_fingerprint = scaler_fingerprint(known.observation_builder)
    for row in records:
        checkpoint = root / row["checkpoint_path"]
        if sha256_file(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash changed: {checkpoint}")
        if row["scaler"] != expected_scaler_fingerprint:
            raise RuntimeError(f"known-period scaler differs for seed {row['seed']}")
        model = PPO.load(checkpoint, env=v9._environment(
            known, spy, "v9_gate_relative_reward", training=False), device="cpu")
        model.policy.set_training_mode(False)
        observations: list[np.ndarray] = []
        before = (parameter_sha256(model), optimizer_sha256(model), expected_scaler)
        def choose(observation, _index):
            observations.append(np.asarray(observation).copy())
            action, _ = model.predict(observation, deterministic=True)
            return int(np.asarray(action).item())
        env = v9._environment(known, spy, "v9_gate_relative_reward", training=False)
        rows, metrics, nav_error = rollout(env, choose, seed=int(row["seed"]))
        trace = v9._action_trace(rows, known, "v9_gate_relative_reward")
        after = (parameter_sha256(model), optimizer_sha256(model),
                 scaler_sha256(known.observation_builder))
        seed_results.append({"seed": row["seed"], "technical_pass": before == after and abs(nav_error) <= 1e-10,
                             "reward_nav_error": nav_error,
                             "responsiveness": responsiveness(model, observations),
                             **metrics, **trace})
    payload = {"status": "PASS" if all(x["technical_pass"] for x in seed_results) else "FAIL",
               "label": "known_period_confirmation_not_untouched_test",
               "created_at_utc": now(), "buy_and_hold": bh,
               "baselines": baselines, "seeds": seed_results,
               "preflight": preflight_result}
    write_new(report, payload)
    manifest = {
        "algorithm": "PPO", "arm": "v9_gate_relative_reward",
        "environment": "victimagent.v9.study._environment",
        "wrappers": ["TradeDecisionEnv(v3_min_hold_five)", "TrendGateWrapper",
                     "RelativeRewardWrapper(training only)"],
        "victimagent_commit": EXPECTED_PIN,
        "ppo_repo_commit": git_commit(root), "config_sha256": sha256_file(config.path),
        "processed_data_sha256": config.raw["shared_contract"]["expected_processed_sha256"],
        "scaler": records[0]["scaler"], "attack_policy": "use_all_5_seeds",
        "checkpoints": [{"seed": r["seed"], "path": r["checkpoint_path"],
                         "sha256": r["checkpoint_sha256"]} for r in records],
        "confirmation_report": report.relative_to(root).as_posix(),
    }
    write_new(manifest_path, manifest)
    return payload


def aggregate(root: Path, config: PPOConfigV9) -> dict[str, Any]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for fold in FOLDS:
            for seed in SEEDS:
                revision = config.raw["formal"]["record_revision"]
                path = root / RECORDS / f"{arm}_{fold}_seed{seed}_{revision}.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                if (row.get("status") != "PASS" or not row.get("formal")
                        or row.get("known_period_accessed") is not False
                        or row.get("actual_timesteps") != EXPECTED_ACTUAL_TIMESTEPS):
                    raise ValueError(f"invalid formal PPO V9 record: {path}")
                records[(arm, fold, seed)] = row
    pairs, paired_folds = [], {}
    for fold in FOLDS:
        fold_rows = []
        for seed in SEEDS:
            control = records[(ARMS[0], fold, seed)]
            treatment = records[(ARMS[1], fold, seed)]
            delta = math.log(treatment["final_wealth"] / control["final_wealth"])
            pair = {"fold": fold, "seed": seed, "control_wealth": control["final_wealth"],
                    "treatment_wealth": treatment["final_wealth"], "delta_log_wealth": delta}
            pairs.append(pair)
            fold_rows.append(delta)
        paired_folds[fold] = {"median_delta_log_wealth": float(np.median(fold_rows)),
                              "positive_delta_count": int(sum(x > 0 for x in fold_rows))}
    metric_names = ("final_wealth", "cagr", "annualized_mean_excess_return",
                    "sharpe_excess_cash", "annualized_volatility", "max_drawdown",
                    "average_exposure", "total_turnover", "total_transaction_cost",
                    "gate_forced_share", "gate_override_share")
    by_arm_fold, baselines = {}, {}
    failures, unresponsive, constant = [], [], []
    for arm in ARMS:
        by_arm_fold[arm] = {}
        for fold in FOLDS:
            rows = [records[(arm, fold, seed)] for seed in SEEDS]
            rel = [r["log_relative_wealth_vs_buy_and_hold"] for r in rows]
            blocked = [float(r["executed_action_distribution"].get("None", 0.0)) for r in rows]
            by_arm_fold[arm][fold] = {
                "median_log_wealth_vs_buy_and_hold": float(np.median(rel)),
                "wins_vs_buy_and_hold": int(sum(x > 0 for x in rel)),
                "median_five_day_rule_blocked_share": float(np.median(blocked)),
                "median_metrics": {name: (None if all(r.get(name) is None for r in rows)
                    else float(np.median([r[name] for r in rows if r.get(name) is not None])))
                    for name in metric_names},
            }
            baselines.setdefault(fold, rows[0]["baselines"])
            for row in rows:
                key = {"arm": row["arm"], "fold": fold, "seed": row["seed"]}
                if not row["technical_pass"]:
                    failures.append(key)
                if not row["state_responsive"]:
                    unresponsive.append(key)
                if row["near_constant_policy"]:
                    constant.append(key)
    dqn_path = root / "victimagent/reports/v9/VALIDATION_DECISION.json"
    dqn = json.loads(dqn_path.read_text(encoding="utf-8"))
    summary = {
        "status": "PASS", "created_at_utc": now(), "known_period_accessed": False,
        "algorithm": "PPO", "record_revision": config.raw["formal"]["record_revision"],
        "contrast": "v9_gate_relative_reward minus v9_trend_gate",
        "arm_fold_summary": by_arm_fold, "baselines": baselines,
        "paired_reward_contrast": {"all_pairs": pairs, "per_fold": paired_folds},
        "mean_of_fold_median_delta_log_wealth": float(np.mean(
            [paired_folds[f]["median_delta_log_wealth"] for f in FOLDS])),
        "technical_failures": failures, "non_state_responsive": unresponsive,
        "near_constant_policy": constant,
        "dqn_descriptive_reference_not_paired": {
            "source": "victimagent/reports/v9/VALIDATION_DECISION.json",
            "scores": dqn["scores"], "buy_and_hold": dqn["buy_and_hold"]},
        "final_arm_preregistered_not_selected": "v9_gate_relative_reward",
    }
    write_new(root / REPORTS / "VALIDATION_SUMMARY.json", summary)
    lines = ["# PPO V9 validation summary", "",
             "Known period accessed: **no**. Final arm was preregistered as "
             "`v9_gate_relative_reward`; validation does not select an arm.", "",
             "| Arm | Fold | Median log(W/B&H) | Wins vs B&H | Median wealth | Sharpe | Max DD | Exposure | Turnover | Cost | Gate forced | Gate override | 5-day blocked |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for fold in FOLDS:
            item, m = by_arm_fold[arm][fold], by_arm_fold[arm][fold]["median_metrics"]
            lines.append(f"| {config.upstream_arm(arm)} | {fold} | {item['median_log_wealth_vs_buy_and_hold']:+.4f} | "
                         f"{item['wins_vs_buy_and_hold']}/5 | {m['final_wealth']:.4f} | {m['sharpe_excess_cash']:.3f} | "
                         f"{m['max_drawdown']:.2%} | {m['average_exposure']:.3f} | {m['total_turnover']:.2f} | "
                         f"{m['total_transaction_cost']:.2f} | {m['gate_forced_share']:.2%} | "
                         f"{m['gate_override_share']:.2%} | {item['median_five_day_rule_blocked_share']:.2%} |")
    lines += ["", "## Paired reward contrast", "",
              "| Fold | Median log(W relative/W NAV) | Positive pairs |",
              "|---|---:|---:|"]
    for fold in FOLDS:
        item = paired_folds[fold]
        lines.append(f"| {fold} | {item['median_delta_log_wealth']:+.4f} | {item['positive_delta_count']}/5 |")
    lines += ["", f"Technical failures: {len(failures)}; non-responsive: {len(unresponsive)}; "
              f"near-constant policies: {len(constant)}.", "",
              "DQN values in the JSON are descriptive references from the pinned victimagent report, not paired PPO comparisons.", ""]
    md = root / REPORTS / "VALIDATION_SUMMARY.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    with md.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return summary

