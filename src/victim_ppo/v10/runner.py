from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
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

from victim_ppo.v9.runner import (EXPECTED_BUY_AND_HOLD, build_model,
                                  git_commit, parameter_sha256, preflight,
                                  responsiveness, write_new)
from .config import (ARMS, BETA, EXPECTED_ACTUAL_TIMESTEPS, FOLDS,
                     REQUESTED_TIMESTEPS, SEEDS, STREAK_CAP, PPOConfigV10)
from .wrappers import PositiveStreakRewardWrapper

RECORDS = Path("experiments/v10_ppo/records")
ATTEMPTS = Path("experiments/v10_ppo/attempts")
REPORTS = Path("reports/v10_ppo")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _environment(bundle, spy, config: PPOConfigV10, arm: str, *, training: bool):
    # Both arms inherit the same V9 trend gate. Reward shaping exists only in
    # training; deterministic evaluation always receives true NAV log reward.
    env = v9._environment(bundle, spy, "v9_trend_gate", training=training)
    if training:
        env = PositiveStreakRewardWrapper(
            env, beta=BETA, streak_cap=STREAK_CAP,
            enabled=config.uses_streak_reward(arm),
        )
    return env


def _train(root: Path, config: PPOConfigV10, bundle, spy, arm: str, seed: int,
           requested_timesteps: int, label: str):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    class DiagnosticsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.losses: list[float] = []
            self.bonus_sum = 0.0
            self.nav_reward_sum = 0.0
            self.bonus_steps = 0
            self.max_streak = 0
            self.reward_steps = 0

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "nav_log_reward" not in info:
                    continue
                bonus = float(info["positive_streak_bonus"])
                self.bonus_sum += bonus
                self.nav_reward_sum += float(info["nav_log_reward"])
                self.bonus_steps += int(bonus > 0.0)
                self.max_streak = max(self.max_streak, int(info["positive_return_streak"]))
                self.reward_steps += 1
            return True

        def _on_rollout_end(self) -> None:
            value = self.model.logger.name_to_value.get("train/loss")
            if value is not None:
                self.losses.append(float(value))

    random.seed(seed)
    np.random.seed(seed)
    env = _environment(bundle, spy, config, arm, training=True)
    if env.observation_space.shape != (20,) or env.action_space.n != 3:
        raise AssertionError("PPO V10 did not receive the shared V9 20-D/3-action interface")
    checkpoint = root / "artifacts/checkpoints/v10_ppo" / f"{label}.zip"
    if checkpoint.exists():
        raise FileExistsError(f"checkpoint exists: {checkpoint}")
    tb = root / "tensorboard/v10_ppo" / label
    log = root / "artifacts/logs/v10_ppo" / label
    log.mkdir(parents=True, exist_ok=False)
    model = build_model(env, config, seed=seed, tensorboard_log=tb)
    model.set_logger(configure(str(log), ["csv", "tensorboard"]))
    before_param = parameter_sha256(model)
    before_scaler = scaler_sha256(bundle.observation_builder)
    callback, started = DiagnosticsCallback(), time.time()
    model.learn(total_timesteps=requested_timesteps, callback=callback,
                reset_num_timesteps=True, progress_bar=False, tb_log_name="run")
    if not callback.losses or not np.isfinite(callback.losses).all():
        raise RuntimeError("PPO V10 training did not produce finite losses")
    if parameter_sha256(model) == before_param:
        raise RuntimeError("PPO V10 parameters did not update")
    if scaler_sha256(bundle.observation_builder) != before_scaler:
        raise RuntimeError("training mutated the fold-train-only scaler")
    if callback.reward_steps != int(model.num_timesteps):
        raise RuntimeError("reward diagnostic count does not match PPO transitions")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    denominator = abs(callback.nav_reward_sum)
    return checkpoint, {
        "train_seconds": round(time.time() - started, 1),
        "requested_timesteps": requested_timesteps,
        "actual_timesteps": int(model.num_timesteps),
        "final_logged_loss": callback.losses[-1],
        "tensorboard_path": tb.relative_to(root).as_posix(),
        "training_log_path": log.relative_to(root).as_posix(),
        "training_reward_diagnostics": {
            "nav_reward_sum": callback.nav_reward_sum,
            "positive_streak_bonus_sum": callback.bonus_sum,
            "bonus_positive_step_share": callback.bonus_steps / callback.reward_steps,
            "maximum_positive_return_streak": callback.max_streak,
            "bonus_to_absolute_nav_reward_sum": (
                callback.bonus_sum / denominator if denominator > 0 else None),
        },
    }


def _evaluate(root: Path, config: PPOConfigV10, checkpoint: Path, bundle, spy,
              arm: str, seed: int, label: str) -> dict[str, Any]:
    from stable_baselines3 import PPO

    env = _environment(bundle, spy, config, arm, training=False)
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
    trace = v9._action_trace(rows, bundle, "v9_trend_gate")
    after = (parameter_sha256(model), optimizer_sha256(model),
             scaler_sha256(bundle.observation_builder))
    response = responsiveness(model, observations)
    trajectory = root / "artifacts/trajectories/v10_ppo" / f"{label}_validation.csv"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    if trajectory.exists():
        raise FileExistsError(f"trajectory exists: {trajectory}")
    pd.DataFrame(rows).to_csv(trajectory, index=False)
    largest = max(trace["agent_action_distribution"].values())
    return {
        "technical_pass": bool(before == after and abs(error) <= 1e-10),
        "evaluation_reward": "true_nav_log_return_unshaped",
        "evaluation_state_unchanged": before == after,
        "reward_nav_error": error,
        "responsiveness": response,
        "state_responsive": response["eligible"],
        "near_constant_policy": bool(largest >= 0.95),
        "trajectory_path": trajectory.relative_to(root).as_posix(),
        "trajectory_sha256": sha256_file(trajectory),
        **trace, **measures,
    }


def run_cell(root: Path, config: PPOConfigV10, *, arm: str, fold_id: str, seed: int,
             requested_timesteps: int, tag: str = "", threads: int = 1) -> dict[str, Any]:
    if arm not in ARMS or fold_id not in FOLDS or seed not in SEEDS:
        raise ValueError("invalid PPO V10 arm, fold, or seed")
    formal = requested_timesteps == REQUESTED_TIMESTEPS
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
        raise FileExistsError(f"PPO V10 cell already attempted: {label}")
    write_new(attempt, {"status": "reserved", "created_at_utc": now(), "arm": arm,
                        "fold": fold_id, "seed": seed, "formal": formal})
    result: dict[str, Any] = {}
    status, error, checkpoint = "FAIL", None, None
    try:
        checkpoint, training = _train(root, config, train, train_spy, arm, seed,
                                      requested_timesteps, label)
        evaluation = _evaluate(root, config, checkpoint, validation, val_spy, arm, seed, label)
        baselines = v9._baselines_with_rule(validation, val_spy)
        bh = baselines["buy_and_hold"]["final_wealth"]
        if not math.isclose(bh, EXPECTED_BUY_AND_HOLD[fold_id], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"{fold_id} buy-and-hold {bh} != frozen V9 value")
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
            "algorithm": "PPO", "experiment": "v10_nav_positive_streak_reward",
            "arm": arm, "fold": fold_id, "seed": seed,
            "total_timesteps": requested_timesteps, "created_at_utc": now(),
            "known_period_accessed": False, "preflight": provenance,
            "hyperparameters": config.model, "reward_contract": config.raw["streak_reward"],
            "streak_reward_enabled": config.uses_streak_reward(arm),
            "scaler": scaler_fingerprint(train.observation_builder),
            "train_window": train.metadata, "validation_window": validation.metadata,
            "checkpoint_path": checkpoint.relative_to(root).as_posix() if checkpoint else None,
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
            **result,
        })
    return json.loads(record_path.read_text(encoding="utf-8"))


def _median(records: list[dict[str, Any]], key: str) -> float:
    return float(np.median([float(row[key]) for row in records]))


def aggregate(root: Path, config: PPOConfigV10) -> dict[str, Any]:
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    revision = config.raw["formal"]["record_revision"]
    for arm in ARMS:
        for fold in FOLDS:
            for seed in SEEDS:
                path = root / RECORDS / f"{arm}_{fold}_seed{seed}_{revision}.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                if (row.get("status") != "PASS" or not row.get("formal")
                        or row.get("known_period_accessed") is not False
                        or row.get("actual_timesteps") != EXPECTED_ACTUAL_TIMESTEPS):
                    raise ValueError(f"invalid formal PPO V10 record: {path}")
                records[(arm, fold, seed)] = row

    pairs, per_fold = [], {}
    positive_pairs = 0
    for fold in FOLDS:
        deltas = []
        for seed in SEEDS:
            control = records[(ARMS[0], fold, seed)]
            treatment = records[(ARMS[1], fold, seed)]
            delta = math.log(treatment["final_wealth"] / control["final_wealth"])
            positive_pairs += int(delta > 0)
            deltas.append(delta)
            pairs.append({"fold": fold, "seed": seed,
                          "control_wealth": control["final_wealth"],
                          "treatment_wealth": treatment["final_wealth"],
                          "delta_log_wealth": delta})
        per_fold[fold] = {"median_delta_log_wealth": float(np.median(deltas)),
                          "positive_delta_count": int(sum(x > 0 for x in deltas))}

    control_rows = [records[(ARMS[0], fold, seed)] for fold in FOLDS for seed in SEEDS]
    treatment_rows = [records[(ARMS[1], fold, seed)] for fold in FOLDS for seed in SEEDS]
    positive_folds = sum(per_fold[f]["median_delta_log_wealth"] > 0 for f in FOLDS)
    drawdown_deterioration_pp = 100 * (
        abs(_median(treatment_rows, "max_drawdown")) - abs(_median(control_rows, "max_drawdown")))
    control_cost = _median(control_rows, "total_transaction_cost")
    cost_increase = (_median(treatment_rows, "total_transaction_cost") / control_cost - 1.0
                     if control_cost > 0 else math.inf)
    rule = config.raw["adoption"]
    adopted = bool(
        positive_folds >= rule["minimum_positive_fold_medians"]
        and positive_pairs >= rule["minimum_positive_pairs"]
        and drawdown_deterioration_pp <= rule["maximum_median_drawdown_deterioration_pp"]
        and cost_increase <= rule["maximum_median_transaction_cost_increase_fraction"]
    )
    summary = {
        "status": "PASS", "created_at_utc": now(), "known_period_accessed": False,
        "algorithm": "PPO", "experiment": "v10_nav_positive_streak_reward",
        "record_revision": revision, "all_pairs": pairs, "per_fold": per_fold,
        "positive_fold_medians": positive_folds, "positive_pairs": positive_pairs,
        "median_drawdown_deterioration_pp": drawdown_deterioration_pp,
        "median_transaction_cost_increase_fraction": cost_increase,
        "adoption_rule": rule,
        "adopt_treatment": adopted,
        "selected_arm": ARMS[1] if adopted else ARMS[0],
        "interpretation": "Development evidence only; evaluation uses unshaped true NAV reward.",
    }
    write_new(root / REPORTS / "VALIDATION_SUMMARY.json", summary)
    lines = [
        "# PPO V10 validation summary", "",
        "Known period accessed: **no**. Evaluation reward: **unshaped true NAV log return**.", "",
        "| Fold | Median log(W streak/W NAV) | Positive pairs |", "|---|---:|---:|",
    ]
    for fold in FOLDS:
        item = per_fold[fold]
        lines.append(f"| {fold} | {item['median_delta_log_wealth']:+.4f} | "
                     f"{item['positive_delta_count']}/5 |")
    lines += ["", f"Positive fold medians: **{positive_folds}/3**", "",
              f"Positive paired runs: **{positive_pairs}/15**", "",
              f"Treatment adopted: **{'yes' if adopted else 'no'}**", "",
              "This is development evidence, not an untouched out-of-sample result.", ""]
    md = root / REPORTS / "VALIDATION_SUMMARY.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    with md.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return summary


__all__ = ["aggregate", "preflight", "run_cell"]

