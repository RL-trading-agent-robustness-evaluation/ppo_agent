from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from victimagent.data.real_adapter import prepare_asset_bundle
from victimagent.data.splits import load_split_contract
from victimagent.evaluation import scaler_fingerprint, scaler_sha256
from victimagent.v2.folds import load_rolling_folds
from victimagent.v2.registry import sha256_file
from victimagent.v8 import study as v8
from victimagent.v9 import study as v9
from victim_ppo.config import load_config as load_v4_config
from victim_ppo.runner import run_evaluation as evaluate_v4, scaler_payload
from victim_ppo.v10.config import load_config as load_v10_config
from victim_ppo.v10.runner import _evaluate as evaluate_v10


LABEL = "known_period_descriptive_comparison_v4_v9_v10_not_final_test"
EXPECTED_START = "2022-01-03"
EXPECTED_END = "2025-12-31"
EXPECTED_STEPS = 1002
EXPECTED_BH = 1.48310441640342
METRICS = (
    "final_wealth", "cagr", "annualized_mean_excess_return",
    "sharpe_excess_cash", "annualized_volatility", "max_drawdown",
    "average_exposure", "total_turnover", "total_transaction_cost",
)


def distribution(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {
        "median": float(np.median(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def verify_boundary(row: dict, label: str) -> None:
    actual = (int(row["steps"]), row["start_date"], row["end_date"])
    expected = (EXPECTED_STEPS, EXPECTED_START, EXPECTED_END)
    if actual != expected:
        raise RuntimeError(f"{label} boundary {actual} != {expected}")


def git_sha(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    victim = root / "victimagent"
    report_dir = root / "reports/v10_ppo/comparison_2022_2025"
    json_path = report_dir / "COMPARISON.json"
    md_path = report_dir / "COMPARISON.md"
    attempt = root / "experiments/ppo_known_comparison_2022_2025/attempt.json"
    for path in (attempt, json_path, md_path):
        if path.exists():
            raise FileExistsError(f"comparison output already exists: {path}")
    attempt.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({
        "status": "reserved", "label": LABEL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "known_period_accessed": True,
    }, indent=2), encoding="utf-8")

    # V9: retain the already-completed one-shot diagnostic; do not rerun it.
    v9_report_path = root / "reports/v9_ppo_known_diagnostic/KNOWN_PERIOD_DIAGNOSTIC.json"
    v9_report = json.loads(v9_report_path.read_text(encoding="utf-8"))
    if (v9_report.get("status") != "PASS" or len(v9_report.get("per_seed", [])) != 5
            or v9_report.get("known_period_accessed") is not True):
        raise RuntimeError("V9 known-period diagnostic is not a complete five-seed PASS")
    v9_rows = v9_report["per_seed"]
    for row in v9_rows:
        verify_boundary(row, f"V9 seed {row['seed']}")
    buy_hold = v9_report["baselines"]["buy_and_hold"]
    if not math.isclose(float(buy_hold["final_wealth"]), EXPECTED_BH,
                        rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("V9 executable buy-and-hold differs from the frozen value")

    # V4: reconstruct the frozen 2010-2018 train-only scaler and select test rows.
    v4_config = load_v4_config(root / "configs/ppo_v4.yaml")
    market_v4 = pd.read_csv(root / "data/processed/market_base_with_splits.csv")
    events_v4 = pd.read_csv(root / "data/interim/distribution_events.csv")
    split_v4 = load_split_contract(victim / "configs/splits.yaml")
    target_v4 = prepare_asset_bundle(
        market_v4, events_v4, split_v4, ticker="SPY", partition="test")
    v4_rows = []
    v4_trajectory_dir = root / "artifacts/trajectories/ppo_known_comparison/v4"
    for seed in range(5):
        source = json.loads((root / f"reports/PPO_CLEAN_SEED{seed}.json").read_text(
            encoding="utf-8"))
        checkpoint = root / source["artifacts"]["checkpoint"]
        scaler_file = root / source["artifacts"]["scaler"]
        if sha256_file(checkpoint) != source["artifacts"]["checkpoint_sha256"]:
            raise RuntimeError(f"V4 seed {seed} checkpoint hash mismatch")
        if sha256_file(scaler_file) != source["artifacts"]["scaler_sha256"]:
            raise RuntimeError(f"V4 seed {seed} scaler-file hash mismatch")
        recorded_scaler = json.loads(scaler_file.read_text(encoding="utf-8"))
        if scaler_payload(target_v4.observation_builder) != recorded_scaler:
            raise RuntimeError(f"V4 seed {seed} reconstructed scaler mismatch")
        rows, metrics, error, frozen = evaluate_v4(
            checkpoint, target_v4, v4_config, seed=seed)
        verify_boundary(metrics, f"V4 seed {seed}")
        trajectory = v4_trajectory_dir / f"seed{seed}.csv"
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        if trajectory.exists():
            raise FileExistsError(f"trajectory already exists: {trajectory}")
        pd.DataFrame(rows).to_csv(trajectory, index=False)
        v4_rows.append({
            "seed": seed, "technical_pass": bool(abs(error) <= 1e-10),
            "reward_nav_identity_error": error, **frozen,
            "checkpoint_path": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "scaler_path": scaler_file.relative_to(root).as_posix(),
            "scaler_file_sha256": sha256_file(scaler_file),
            "trajectory_path": trajectory.relative_to(root).as_posix(),
            "trajectory_sha256": sha256_file(trajectory), **metrics,
        })

    # V10: retain every fold-1 pilot seed and reconstruct its fold-train-only scaler.
    v10_config = load_v10_config(root / "configs/ppo_v10_nav_trend_reward.yaml")
    folds = {fold.fold_id: fold for fold in load_rolling_folds(
        victim / "configs/v8_folds.yaml").folds}
    market_v10, events_v10 = v8._v8_inputs(victim, victim, before=None)
    target_v10, spy_v10 = v9.v9_bundle(
        market_v10, events_v10, fit=folds["fold_1"].train, select=v8.KNOWN)
    v10_rows = []
    for seed in range(5):
        record_path = root / "experiments/v10_ppo/records" / (
            f"ppo_v10_gate_nav_streak_fold_1_seed{seed}_pilot200k_v1.json")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record.get("status") != "PASS" or record.get("seed") != seed
                or record.get("actual_timesteps") != 200_704):
            raise RuntimeError(f"V10 seed {seed} is not the expected pilot PASS")
        if record["scaler"] != scaler_fingerprint(target_v10.observation_builder):
            raise RuntimeError(f"V10 seed {seed} reconstructed scaler mismatch")
        checkpoint = root / record["checkpoint_path"]
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError(f"V10 seed {seed} checkpoint hash mismatch")
        label = f"comparison_2022_2025_v10_streak_seed{seed}"
        result = evaluate_v10(root, v10_config, checkpoint, target_v10, spy_v10,
                              "ppo_v10_gate_nav_streak", seed, label)
        verify_boundary(result, f"V10 seed {seed}")
        v10_rows.append({"seed": seed, "record_path": record_path.relative_to(root).as_posix(),
                         "checkpoint_sha256": record["checkpoint_sha256"],
                         "scaler_sha256": scaler_sha256(target_v10.observation_builder), **result})

    groups = {"v4": v4_rows, "v9_relative_fold3": v9_rows, "v10_streak_pilot": v10_rows}
    if not all(row.get("technical_pass", row.get("status") == "PASS")
               for rows in groups.values() for row in rows):
        raise RuntimeError("one or more model evaluations failed technical checks")
    distributions = {
        name: {metric: distribution(rows, metric) for metric in METRICS}
        for name, rows in groups.items()
    }
    wins = {name: int(sum(float(row["final_wealth"]) > EXPECTED_BH for row in rows))
            for name, rows in groups.items()}
    payload = {
        "status": "PASS", "label": LABEL, "known_period_accessed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outer_evaluation_commit": git_sha(root),
        "victimagent_commit": git_sha(victim),
        "data_sha256": {
            "v4_market": sha256_file(root / "data/processed/market_base_with_splits.csv"),
            "v4_distributions": sha256_file(root / "data/interim/distribution_events.csv"),
            "v9_v10_market": v9_report["processed_data_sha256"]["market"],
            "v9_v10_distributions": v9_report["processed_data_sha256"]["events"],
        },
        "evaluation_window": {"start": EXPECTED_START, "end": EXPECTED_END,
                              "steps": EXPECTED_STEPS},
        "comparison_caveats": [
            "All 2022-2025 results are known-period diagnostics, not untouched tests.",
            "V4 uses 10 features, a 2010-2018 training window, 200k steps, and its V4 execution contract.",
            "V9 uses fold 3, a 1994-2018 training window, 500k requested steps, and the V9 gate/min-hold contract.",
            "V10 is a 200k fold-1 development pilot trained through 2014 with the V9 environment and streak-shaped training reward.",
            "The comparison is descriptive and must not be used for model, seed, checkpoint, or reward selection.",
        ],
        "spy_buy_and_hold": buy_hold,
        "distribution": distributions, "wins_vs_buy_and_hold": wins,
        "per_seed": groups,
        "source_v9_report": v9_report_path.relative_to(root).as_posix(),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# PPO V4 / V9 / V10：2022–2025 描述性比較", "",
             "> 本比較使用已開封的 2022–2025 known period，不是 untouched test，也不是最終模型確認。"
             "模型的訓練窗口、特徵、訓練步數與執行規則不同，因此只能作描述性比較，不可據此選模型、seed、checkpoint 或 reward。",
             "", f"固定標籤：`{LABEL}`", "",
             f"共同評估期間：{EXPECTED_START} 至 {EXPECTED_END}，{EXPECTED_STEPS} steps。", "",
             "所有數值均為五個 seeds 的中位數。", "",
             "| Model | Final wealth | CAGR | Excess return | Sharpe | Volatility | Max DD | Exposure | Turnover | Cost | Wins vs B&H |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    display = (("V4", "v4"), ("V9 relative fold 3", "v9_relative_fold3"),
               ("V10 streak pilot", "v10_streak_pilot"))
    for label, key in display:
        d = distributions[key]
        lines.append(f"| {label} | {d['final_wealth']['median']:.4f}× | "
                     f"{d['cagr']['median']:.2%} | {d['annualized_mean_excess_return']['median']:.2%} | "
                     f"{d['sharpe_excess_cash']['median']:.3f} | {d['annualized_volatility']['median']:.2%} | "
                     f"{d['max_drawdown']['median']:.2%} | {d['average_exposure']['median']:.2%} | "
                     f"{d['total_turnover']['median']:.2f}× | ${d['total_transaction_cost']['median']:,.0f} | {wins[key]}/5 |")
    lines.append(f"| SPY buy-and-hold | {buy_hold['final_wealth']:.4f}× | {buy_hold['cagr']:.2%} | "
                 f"{buy_hold['annualized_mean_excess_return']:.2%} | {buy_hold['sharpe_excess_cash']:.3f} | "
                 f"{buy_hold['annualized_volatility']:.2%} | {buy_hold['max_drawdown']:.2%} | "
                 f"{buy_hold['average_exposure']:.2%} | {buy_hold['total_turnover']:.2f}× | "
                 f"${buy_hold['total_transaction_cost']:,.0f} | — |")
    lines += ["", "## Comparability limits", ""] + [
        f"- {item}" for item in payload["comparison_caveats"]]
    lines += ["", "完整 per-seed 成績、IQR、min/max、hash 與 technical checks 請見 JSON。", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "median": {
        name: distributions[name]["final_wealth"]["median"] for name in groups},
        "wins_vs_buy_and_hold": wins}, indent=2))


if __name__ == "__main__":
    main()
