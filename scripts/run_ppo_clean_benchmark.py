from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(path: Path) -> str:
    return subprocess.run(["git", "-c", f"safe.directory={path.resolve().as_posix()}",
                           "-C", str(path), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def write_report(root: Path, seed: int, report: dict, *, smoke: bool) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stem = reports / (f"PPO_SMOKE_SEED{seed}" if smoke else f"PPO_CLEAN_SEED{seed}")
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                          encoding="utf-8")
    m = report["validation"]
    sharpe = "NA" if m["sharpe_excess_cash"] is None else f"{m['sharpe_excess_cash']:.4f}"
    text = [
        f"# PPO V4 {'smoke' if smoke else 'clean benchmark'} — seed {seed}", "",
        f"- Victimagent V4 commit: `{report['provenance']['victimagent_commit']}`",
        f"- Test/known period accessed: **{report['test_partition_accessed']}**",
        f"- Training steps: **{report['training']['total_timesteps']}**", "",
        "## Deterministic 2019–2021 validation", "",
        f"- Final wealth: {m['final_wealth']:.6f}×",
        f"- CAGR: {m['cagr']:.4%}",
        f"- Annualized excess return versus lagged cash: {m['annualized_mean_excess_return']:.4%}",
        f"- Annualized volatility: {m['annualized_volatility']:.4%}",
        f"- Sharpe versus lagged cash: {sharpe}",
        f"- Maximum drawdown: {m['max_drawdown']:.4%}",
        f"- Total turnover: {m['total_turnover']:.6f}",
        f"- Total transaction cost: {m['total_transaction_cost']:.2f}",
        f"- Action distribution: `{m['action_distribution']}`",
        f"- Reward–NAV identity error: `{report['reward_nav_invariant_error']}`", "",
        "This is a validation report, not untouched-test evidence.", "",
    ]
    stem.with_suffix(".md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one PPO seed on shared victimagent V4")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--smoke", action="store_true",
                        help="Run a non-formal 4096-step integration smoke test")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Smoke-only step override; forbidden for formal runs")
    args = parser.parse_args()
    root, data_root = args.root.resolve(), args.data_root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "victimagent/src")]

    from victim_ppo.config import V4_COMMIT, load_config
    from victim_ppo.data import load_development_bundles
    from victim_ppo.runner import run_evaluation, run_training

    config_path = root / "configs/ppo_v4.yaml"
    config = load_config(config_path)
    if args.seed not in config.formal["seeds"]:
        raise ValueError("seed must be one of the preregistered seeds 0..4")
    if args.timesteps is not None and not args.smoke:
        raise ValueError("--timesteps is smoke-only; formal budget cannot be overridden")
    submodule_sha = git_sha(root / "victimagent")
    if submodule_sha != V4_COMMIT:
        raise RuntimeError(f"victimagent must be pinned to {V4_COMMIT}, got {submodule_sha}")
    if not args.smoke:
        if config.raw["status"] != "frozen_before_ppo_training":
            raise RuntimeError("formal PPO configuration is not frozen")
        dirty = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root),
             "status", "--porcelain", "--untracked-files=normal", "--",
             "src", "scripts", "configs", "pyproject.toml", "victimagent"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError("formal PPO training requires a clean committed worktree")
    timesteps = (args.timesteps or 4096) if args.smoke else int(
        config.formal["total_timesteps_per_seed"])
    train, validation = load_development_bundles(root, data_root)
    label = "ppo_v4_smoke" if args.smoke else "ppo_v4"
    artifact = root / f"artifacts/checkpoints/{label}_seed{args.seed}"
    if not args.smoke and artifact.with_suffix(".zip").exists():
        raise FileExistsError(f"formal checkpoint already exists: {artifact.with_suffix('.zip')}")
    model, scaler_path = run_training(
        train, config, seed=args.seed, timesteps=timesteps, checkpoint=artifact,
        log_dir=root / f"artifacts/logs/{label}_seed{args.seed}")
    del model
    rows, metrics, error, frozen = run_evaluation(
        artifact.with_suffix(".zip"), validation, config, seed=args.seed)
    trajectory = root / f"artifacts/trajectories/{label}_seed{args.seed}_validation.csv"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(trajectory, index=False)
    packages = {}
    for package in ("stable-baselines3", "torch", "numpy", "gymnasium", "pandas"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    inputs = {
        "market": data_root / "data/processed/market_base_with_splits.csv",
        "distributions": data_root / "data/interim/distribution_events.csv",
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "PPO", "seed": args.seed,
        "test_partition_accessed": False, "known_period_accessed": False,
        "run_type": "smoke" if args.smoke else "formal",
        "training": {"partition": "train", "total_timesteps": timesteps,
                     "formal_budget": timesteps == 200_000},
        "validation": metrics, "reward_nav_invariant_error": error,
        "frozen_evaluation": frozen,
        "artifacts": {"checkpoint": str(artifact.with_suffix('.zip').relative_to(root)),
                      "scaler": str(scaler_path.relative_to(root)),
                      "trajectory": str(trajectory.relative_to(root))},
        "provenance": {
            "outer_commit": git_sha(root), "victimagent_commit": submodule_sha,
            "config_sha256": sha256_file(config_path),
            "split_config_sha256": sha256_file(root / "victimagent/configs/splits.yaml"),
            "data_sha256": {name: sha256_file(path) for name, path in inputs.items()},
            "python": sys.version.split()[0], "packages": packages,
        },
    }
    write_report(root, args.seed, report, smoke=args.smoke)


if __name__ == "__main__":
    main()

