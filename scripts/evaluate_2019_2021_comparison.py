from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from victimagent.v2.folds import load_rolling_folds
from victimagent.v8 import study as v8
from victimagent.v9 import study as v9
from victim_ppo.v10.config import load_config
from victim_ppo.v10.runner import _evaluate


def median_rows(rows, keys):
    return {key: float(np.median([float(row[key]) for row in rows])) for key in keys}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    victim = root / "victimagent"
    config = load_config(root / "configs/ppo_v10_nav_trend_reward.yaml")
    folds = {fold.fold_id: fold for fold in load_rolling_folds(victim / "configs/v8_folds.yaml").folds}
    market, events = v8._v8_inputs(victim, victim)
    target, spy = v9.v9_bundle(
        market, events, fit=folds["fold_1"].train, select=folds["fold_3"].validation)

    v10_rows = []
    for seed in range(5):
        checkpoint = root / "artifacts/checkpoints/v10_ppo" / (
            f"ppo_v10_gate_nav_streak_fold_1_seed{seed}_pilot200k_v1.zip")
        label = f"comparison_2019_2021_v10_streak_seed{seed}"
        trajectory = root / "artifacts/trajectories/v10_ppo" / f"{label}_validation.csv"
        if trajectory.exists():
            raise FileExistsError(f"comparison trajectory already exists: {trajectory}")
        result = _evaluate(root, config, checkpoint, target, spy,
                           "ppo_v10_gate_nav_streak", seed, label)
        v10_rows.append({"seed": seed, **result})

    v9_rows = {}
    for label, arm in (("v9_control", "ppo_v9_gate_nav"),
                       ("v9_relative", "ppo_v9_gate_relative")):
        v9_rows[label] = [json.loads((root / "experiments/v9_ppo/records" /
            f"{arm}_fold_3_seed{seed}_v9spec1.json").read_text(encoding="utf-8"))
            for seed in range(5)]
    v4 = json.loads((root / "reports/PPO_CLEAN_BENCHMARK.json").read_text(encoding="utf-8"))
    v4_rows = v4["per_seed"]
    metrics = ("final_wealth", "cagr", "sharpe_excess_cash", "annualized_volatility",
               "max_drawdown", "average_exposure", "total_turnover", "total_transaction_cost")
    baselines = v9_rows["v9_control"][0]["baselines"]
    summary = {
        "status": "PASS" if all(row["technical_pass"] for row in v10_rows) else "FAIL",
        "period": "2019-01-02 through 2021-12-31",
        "known_period_accessed": False,
        "comparison_caveats": [
            "V4 uses the historical 10-feature V4 protocol and 200k steps.",
            "V9 control and relative use the 20-feature V9 fold-3 protocol, train through 2018, and 500k steps.",
            "V10 streak uses the 20-feature V9 environment, its fold-1 scaler, train through 2014, and 200k steps.",
            "This is a descriptive common-period comparison, not a matched causal reward ablation.",
        ],
        "spy_buy_and_hold": {key: baselines["buy_and_hold"].get(key) for key in metrics},
        "median_metrics": {
            "v4": median_rows(v4_rows, metrics),
            "v9_control": median_rows(v9_rows["v9_control"], metrics),
            "v9_relative": median_rows(v9_rows["v9_relative"], metrics),
            "v10_streak": median_rows(v10_rows, metrics),
        },
        "per_seed": {
            "v4": [{"seed": row["seed"], **{k: row[k] for k in metrics}} for row in v4_rows],
            "v9_control": [{"seed": row["seed"], **{k: row[k] for k in metrics}} for row in v9_rows["v9_control"]],
            "v9_relative": [{"seed": row["seed"], **{k: row[k] for k in metrics}} for row in v9_rows["v9_relative"]],
            "v10_streak": [{"seed": row["seed"], **{k: row[k] for k in metrics}} for row in v10_rows],
        },
    }
    report_dir = root / "reports/v10_ppo/comparison_2019_2021"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "COMPARISON.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# PPO model comparison, 2019–2021", "",
             "All figures are medians across five seeds. Evaluation uses true NAV and 10 bps costs.", "",
             "| Model | Final wealth | CAGR | Sharpe | Volatility | Max drawdown | Exposure | Turnover | Cost |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("v4", "v9_control", "v9_relative", "v10_streak"):
        row = summary["median_metrics"][name]
        lines.append(f"| {name} | {row['final_wealth']:.4f}× | {row['cagr']:.2%} | "
                     f"{row['sharpe_excess_cash']:.3f} | {row['annualized_volatility']:.2%} | "
                     f"{row['max_drawdown']:.2%} | {row['average_exposure']:.2%} | "
                     f"{row['total_turnover']:.2f}× | ${row['total_transaction_cost']:,.0f} |")
    spy_row = summary["spy_buy_and_hold"]
    lines.append(f"| SPY buy-and-hold | {spy_row['final_wealth']:.4f}× | {spy_row['cagr']:.2%} | "
                 f"{spy_row['sharpe_excess_cash']:.3f} | {spy_row['annualized_volatility']:.2%} | "
                 f"{spy_row['max_drawdown']:.2%} | {spy_row['average_exposure']:.2%} | "
                 f"{spy_row['total_turnover']:.2f}× | ${spy_row['total_transaction_cost']:,.0f} |")
    lines += ["", "## Comparability", ""] + [f"- {item}" for item in summary["comparison_caveats"]]
    (report_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

