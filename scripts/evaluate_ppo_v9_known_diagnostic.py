#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from victimagent.evaluation import optimizer_sha256, scaler_fingerprint, scaler_sha256
from victimagent.v2.folds import load_rolling_folds
from victimagent.v2.registry import sha256_file
from victimagent.v4.evaluation import rollout
from victimagent.v8 import study as v8
from victimagent.v9 import study as v9
from victim_ppo.v9.runner import parameter_sha256, responsiveness, write_new

LABEL = "known_period_diagnostic_fold3_not_final_test"
EXPECTED_PIN = "f4988db5c98d9248533c4ca852cc48ca3f6b7aad"
EXPECTED_MARKET = "281ed8c0530720b5cedf4abd7e9288375dad4fe073f55b0b814560ebcb54af9b"
EXPECTED_EVENTS = "a981523e00b05e804091bf4671835bade363559215403614344bf5a099541668"
EXPECTED_BUY_HOLD = 1.48310441640342
EXPECTED_STEPS = 1002
EXPECTED_START = "2022-01-03"
EXPECTED_END = "2025-12-31"
SEEDS = tuple(range(5))
METRICS = (
    "final_wealth", "cagr", "annualized_mean_excess_return",
    "sharpe_excess_cash", "annualized_volatility", "max_drawdown",
    "average_exposure", "total_turnover", "total_transaction_cost",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_sha(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    victim = root / "victimagent"
    attempt = root / "experiments/v9_ppo_known_diagnostic/attempt.json"
    report_dir = root / "reports/v9_ppo_known_diagnostic"
    json_report = report_dir / "KNOWN_PERIOD_DIAGNOSTIC.json"
    md_report = report_dir / "KNOWN_PERIOD_DIAGNOSTIC.md"
    error_report = root / "experiments/v9_ppo_known_diagnostic/error.json"
    for path in (attempt, json_report, md_report):
        if path.exists():
            raise FileExistsError(f"diagnostic output already exists: {path}")
    write_new(attempt, {"status": "reserved", "label": LABEL,
                        "created_at_utc": now(), "known_period_accessed": False})
    try:
        outer_sha, pin = git_sha(root), git_sha(victim)
        if pin != EXPECTED_PIN:
            raise RuntimeError(f"victimagent pin {pin} != {EXPECTED_PIN}")
        config_path = root / "configs/ppo_v9_reward_ablation.yaml"
        config_hash = sha256_file(config_path)
        folds = {f.fold_id: f for f in load_rolling_folds(victim / "configs/v8_folds.yaml").folds}
        market, events = v8._v8_inputs(victim, victim, before=None)
        bundle, spy = v9.v9_bundle(market, events, fit=folds["fold_3"].train,
                                   select=v8.KNOWN)
        scaler = scaler_fingerprint(bundle.observation_builder)
        scaler_hash = scaler_sha256(bundle.observation_builder)
        data_build = json.loads((victim / "reports/v8/DATA_BUILD.json").read_text(encoding="utf-8"))
        processed = data_build["processed_sha256"]
        market_key = "data/processed/v8/spy_market_1993_2025.csv"
        event_key = "data/interim/v8/spy_cash_distributions_1993_2025.csv"
        if processed.get(market_key) != EXPECTED_MARKET or processed.get(event_key) != EXPECTED_EVENTS:
            raise RuntimeError("processed data hashes do not match the frozen contract")

        records = []
        for seed in SEEDS:
            path = root / "experiments/v9_ppo/records" / (
                f"ppo_v9_gate_relative_fold_3_seed{seed}_v9spec1.json")
            record = json.loads(path.read_text(encoding="utf-8"))
            if (record.get("status") != "PASS" or record.get("formal") is not True
                    or record.get("fold") != "fold_3" or record.get("seed") != seed
                    or record.get("actual_timesteps") != 501_760
                    or record.get("known_period_accessed") is not False):
                raise RuntimeError(f"ineligible original record: {path}")
            if record["scaler"] != scaler:
                raise RuntimeError(f"fold-3 scaler mismatch for seed {seed}")
            checkpoint = root / record["checkpoint_path"]
            if sha256_file(checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch for seed {seed}")
            records.append((record, checkpoint))

        baselines = v9._baselines_with_rule(bundle, spy)
        buy_hold = baselines["buy_and_hold"]
        if not math.isclose(float(buy_hold["final_wealth"]), EXPECTED_BUY_HOLD,
                            rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("known-period buy-and-hold does not match frozen value")
        if (int(buy_hold["steps"]) != EXPECTED_STEPS
                or buy_hold["start_date"] != EXPECTED_START
                or buy_hold["end_date"] != EXPECTED_END):
            raise RuntimeError("known-period boundary differs from the team contract")

        seed_results = []
        trajectory_dir = root / "artifacts/trajectories/v9_ppo_known_diagnostic"
        for record, checkpoint in records:
            seed = int(record["seed"])
            env = v9._environment(bundle, spy, "v9_gate_relative_reward", training=False)
            model = PPO.load(checkpoint, env=env, device="cpu")
            model.policy.set_training_mode(False)
            before = (parameter_sha256(model), optimizer_sha256(model), scaler_hash)
            observations: list[np.ndarray] = []

            def choose(observation, _index):
                observations.append(np.asarray(observation).copy())
                action, _ = model.predict(observation, deterministic=True)
                return int(np.asarray(action).item())

            rows, metrics, reward_error = rollout(env, choose, seed=seed)
            trace = v9._action_trace(rows, bundle, "v9_gate_relative_reward")
            after = (parameter_sha256(model), optimizer_sha256(model),
                     scaler_sha256(bundle.observation_builder))
            if (metrics["steps"] != EXPECTED_STEPS or metrics["start_date"] != EXPECTED_START
                    or metrics["end_date"] != EXPECTED_END):
                raise RuntimeError(f"seed {seed} evaluation boundary mismatch")
            trajectory = trajectory_dir / f"seed{seed}.csv"
            trajectory.parent.mkdir(parents=True, exist_ok=True)
            if trajectory.exists():
                raise FileExistsError(f"trajectory exists: {trajectory}")
            pd.DataFrame(rows).to_csv(trajectory, index=False)
            technical_pass = bool(before == after and abs(reward_error) <= 1e-10)
            if not technical_pass:
                raise RuntimeError(f"technical evaluation failure for seed {seed}")
            seed_results.append({
                "seed": seed,
                "status": "PASS",
                "training_outer_commit": record["preflight"]["ppo_repo_commit"],
                "record_path": (root / "experiments/v9_ppo/records" /
                    f"ppo_v9_gate_relative_fold_3_seed{seed}_v9spec1.json").relative_to(root).as_posix(),
                "checkpoint_path": record["checkpoint_path"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "scaler_sha256": scaler_hash,
                "trajectory_path": trajectory.relative_to(root).as_posix(),
                "trajectory_sha256": sha256_file(trajectory),
                "model_optimizer_scaler_unchanged": before == after,
                "reward_nav_identity_error": reward_error,
                "nav_finite_positive": True,
                "natural_termination": True,
                "responsiveness": responsiveness(model, observations),
                "log_wealth_gap_vs_buy_and_hold": math.log(
                    float(metrics["final_wealth"]) / float(buy_hold["final_wealth"])),
                "min_hold_blocked_share": float(trace["executed_action_distribution"].get("None", 0.0)),
                **trace,
                **metrics,
            })

        summary = {metric: distribution([float(row[metric]) for row in seed_results])
                   for metric in METRICS}
        gaps = [float(row["log_wealth_gap_vs_buy_and_hold"]) for row in seed_results]
        summary["log_wealth_gap_vs_buy_and_hold"] = distribution(gaps)
        payload: dict[str, Any] = {
            "status": "PASS",
            "label": LABEL,
            "known_period_accessed": True,
            "created_at_utc": now(),
            "outer_evaluation_commit": outer_sha,
            "victimagent_commit": pin,
            "original_training_commits": sorted({
                row["training_outer_commit"] for row in seed_results}),
            "config_path": config_path.relative_to(root).as_posix(),
            "config_sha256_at_evaluation": config_hash,
            "original_record_config_sha256": sorted({
                record["preflight"]["ppo_config_sha256"] for record, _ in records}),
            "processed_data_sha256": {"market": EXPECTED_MARKET, "events": EXPECTED_EVENTS},
            "scaler": scaler,
            "scaler_sha256": scaler_hash,
            "training_window": {"start": "1994-02-01", "end": "2018-12-31"},
            "original_validation_window": {"start": "2019-01-01", "end": "2021-12-31"},
            "evaluation_window": {"start": EXPECTED_START, "end": EXPECTED_END,
                                  "steps": EXPECTED_STEPS},
            "evaluation_contract": {
                "environment": "victimagent.v9.study._environment",
                "arm": "v9_gate_relative_reward",
                "training": False,
                "deterministic": True,
                "evaluation_reward": "true after-cost NAV log return",
                "relative_training_reward_applied": False,
                "trend_gate": True,
                "execution_rule": "v3_min_hold_five",
                "action_weights": [0.0, 0.5, 1.0],
                "transaction_cost_bps": 10.0,
                "initial_nav": 1_000_000.0,
            },
            "package_versions": {name: version(dist) for name, dist in {
                "stable_baselines3": "stable-baselines3", "torch": "torch",
                "numpy": "numpy", "gymnasium": "gymnasium", "pandas": "pandas"}.items()},
            "baselines": baselines,
            "per_seed": seed_results,
            "distribution": summary,
            "wins_vs_buy_and_hold": int(sum(gap > 0 for gap in gaps)),
            "interpretation": (
                "Known-period diagnostic using fold-3 checkpoints trained through 2018; "
                "not an untouched test or final-model confirmation."),
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        write_new(json_report, payload)
        intro = ("> 本結果使用 PPO V9 fold-3 checkpoints（train 截至 2018），在已開封 2022–2025 做延伸診斷。"
                 "它不是 untouched test，也不是 train 截至 2021 的最終模型確認。各模型訓練窗口不同的比較僅為描述性比較。")
        lines = ["# PPO V9 2022–2025 known-period diagnostic", "", intro, "",
                 f"固定標籤：`{LABEL}`", "",
                 f"實際期間：{EXPECTED_START} 至 {EXPECTED_END}，{EXPECTED_STEPS} steps", "",
                 "| Seed | Final wealth | CAGR | Excess return | Sharpe | Volatility | Max DD | Exposure | Turnover | Cost | log gap vs B&H |",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in seed_results:
            lines.append(f"| {row['seed']} | {row['final_wealth']:.4f}× | {row['cagr']:.2%} | "
                         f"{row['annualized_mean_excess_return']:.2%} | {row['sharpe_excess_cash']:.3f} | "
                         f"{row['annualized_volatility']:.2%} | {row['max_drawdown']:.2%} | "
                         f"{row['average_exposure']:.2%} | {row['total_turnover']:.2f}× | "
                         f"${row['total_transaction_cost']:,.0f} | {row['log_wealth_gap_vs_buy_and_hold']:+.4f} |")
        lines += ["", f"Wins versus executable SPY buy-and-hold: **{payload['wins_vs_buy_and_hold']}/5**", "",
                  "## Five-seed distribution", "",
                  "| Metric | Median | IQR | Min | Max |", "|---|---:|---:|---:|---:|"]
        for metric, item in summary.items():
            lines.append(f"| {metric} | {item['median']:.6f} | {item['iqr']:.6f} | "
                         f"{item['min']:.6f} | {item['max']:.6f} |")
        lines += ["", "## Technical checks", "",
                  "- All five original records: formal PASS, 501,760 actual steps.",
                  "- All checkpoint, scaler, config, processed-data and trajectory hashes are in the JSON.",
                  "- Model parameters, optimizer and scaler remained unchanged for every rollout.",
                  "- Reward–NAV identity passed within `1e-10`; evaluation used true NAV reward.",
                  f"- Executable buy-and-hold final wealth: `{buy_hold['final_wealth']}`.", "",
                  "## Reproduction", "", "```powershell",
                  ".\\.venv\\Scripts\\python.exe scripts\\evaluate_ppo_v9_known_diagnostic.py", "```", ""]
        with md_report.open("x", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        print(json.dumps(payload, indent=2, allow_nan=False))
    except BaseException as exc:
        write_new(error_report, {"status": "FAIL", "label": LABEL,
                                 "created_at_utc": now(),
                                 "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()

