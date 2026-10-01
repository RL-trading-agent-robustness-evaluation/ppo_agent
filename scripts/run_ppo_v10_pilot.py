#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from victim_ppo.v10.config import ARMS, SEEDS, load_config
from victim_ppo.v10.runner import run_cell


def load_pilot(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "version": 1,
        "status": "frozen_before_ppo_v10_pilot",
        "fold": "fold_1",
        "requested_timesteps_per_run": 200_000,
        "expected_sb3_timesteps_per_run": 200_704,
        "record_tag": "pilot200k_v1",
        "known_period_accessed": False,
        "selection_effect": "none",
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"V10 pilot contract changed: {key}")
    if tuple(raw["arms"]) != ARMS or tuple(raw["seeds"]) != SEEDS:
        raise ValueError("V10 pilot arms or seeds changed")
    return raw


def aggregate(root: Path, pilot: dict) -> dict:
    records = {}
    tag = pilot["record_tag"]
    for arm in ARMS:
        for seed in SEEDS:
            path = root / "experiments/v10_ppo/records" / (
                f"{arm}_{pilot['fold']}_seed{seed}_{tag}.json")
            row = json.loads(path.read_text(encoding="utf-8"))
            if (row.get("status") != "PASS" or row.get("known_period_accessed") is not False
                    or row.get("actual_timesteps") != pilot["expected_sb3_timesteps_per_run"]):
                raise ValueError(f"invalid V10 pilot record: {path}")
            records[(arm, seed)] = row

    pairs = []
    for seed in SEEDS:
        control, treatment = records[(ARMS[0], seed)], records[(ARMS[1], seed)]
        delta = math.log(treatment["final_wealth"] / control["final_wealth"])
        pairs.append({
            "seed": seed,
            "control_final_wealth": control["final_wealth"],
            "treatment_final_wealth": treatment["final_wealth"],
            "delta_log_wealth": delta,
            "control_sharpe": control["sharpe_excess_cash"],
            "treatment_sharpe": treatment["sharpe_excess_cash"],
            "control_max_drawdown": control["max_drawdown"],
            "treatment_max_drawdown": treatment["max_drawdown"],
            "control_turnover": control["total_turnover"],
            "treatment_turnover": treatment["total_turnover"],
            "control_cost": control["total_transaction_cost"],
            "treatment_cost": treatment["total_transaction_cost"],
        })
    deltas = [row["delta_log_wealth"] for row in pairs]
    control_rows = [records[(ARMS[0], seed)] for seed in SEEDS]
    treatment_rows = [records[(ARMS[1], seed)] for seed in SEEDS]
    metrics = ("final_wealth", "cagr", "sharpe_excess_cash", "max_drawdown",
               "average_exposure", "total_turnover", "total_transaction_cost")
    medians = {
        arm: {metric: float(np.median([row[metric] for row in rows]))
              for metric in metrics}
        for arm, rows in ((ARMS[0], control_rows), (ARMS[1], treatment_rows))
    }
    result = {
        "status": "PASS",
        "label": "development_pilot_not_formal_selection",
        "fold": pilot["fold"],
        "requested_timesteps_per_run": pilot["requested_timesteps_per_run"],
        "actual_timesteps_per_run": pilot["expected_sb3_timesteps_per_run"],
        "known_period_accessed": False,
        "pairs": pairs,
        "positive_pairs": int(sum(value > 0 for value in deltas)),
        "median_delta_log_wealth": float(np.median(deltas)),
        "mean_delta_log_wealth": float(np.mean(deltas)),
        "arm_medians": medians,
        "pilot_supports_improvement": bool(np.median(deltas) > 0),
        "caveat": "One-fold 200k development pilot; it does not alter the frozen V10 adoption rule.",
    }
    report_dir = root / "reports/v10_ppo/pilot200k_v1"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "PILOT_SUMMARY.json"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# PPO V10 200k paired pilot", "",
        "Development fold: `fold_1`; known period accessed: **no**.", "",
        "| Seed | NAV wealth | Streak wealth | Delta log wealth | NAV Sharpe | Streak Sharpe |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pairs:
        lines.append(f"| {row['seed']} | {row['control_final_wealth']:.4f} | "
                     f"{row['treatment_final_wealth']:.4f} | {row['delta_log_wealth']:+.4f} | "
                     f"{row['control_sharpe']:.3f} | {row['treatment_sharpe']:.3f} |")
    lines += ["", f"Positive paired seeds: **{result['positive_pairs']}/5**", "",
              f"Median delta log wealth: **{result['median_delta_log_wealth']:+.4f}**", "",
              f"Pilot supports improvement: **{'yes' if result['pilot_supports_improvement'] else 'no'}**", "",
              "This one-fold 200k pilot is diagnostic and does not change the frozen V10 adoption rule.", ""]
    (report_dir / "PILOT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO V10 paired 200k pilot")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    pilot = load_pilot(root / "configs/ppo_v10_pilot_200k.yaml")
    config = load_config(root / "configs/ppo_v10_nav_trend_reward.yaml")
    if args.run:
        if args.arm is None or args.seed is None:
            parser.error("--run requires --arm and --seed")
        result = run_cell(
            root, config, arm=args.arm, fold_id=pilot["fold"], seed=args.seed,
            requested_timesteps=pilot["requested_timesteps_per_run"],
            tag=f"_{pilot['record_tag']}", threads=args.threads,
        )
    else:
        result = aggregate(root, pilot)
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()

