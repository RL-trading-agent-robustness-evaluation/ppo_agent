#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from victim_ppo.v10.config import ARMS, FOLDS, SEEDS, load_config
from victim_ppo.v10.runner import aggregate, preflight, run_cell


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinned PPO V10 NAV plus streak-reward runner")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tag", default="_smoke")
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_config(root / "configs/ppo_v10_nav_trend_reward.yaml")
    if args.preflight:
        result = preflight(root, config, require_data=True)
    elif args.smoke:
        if not args.arm:
            parser.error("--smoke requires --arm")
        spec = config.raw["smoke"]
        result = run_cell(root, config, arm=args.arm, fold_id=spec["fold"],
                          seed=spec["seed"], requested_timesteps=spec["requested_timesteps"],
                          tag=args.tag, threads=args.threads)
    elif args.run:
        if args.arm is None or args.fold is None or args.seed is None:
            parser.error("--run requires --arm, --fold, and --seed")
        formal = config.raw["formal"]
        result = run_cell(root, config, arm=args.arm, fold_id=args.fold, seed=args.seed,
                          requested_timesteps=formal["requested_timesteps_per_run"],
                          tag=f"_{formal['record_revision']}", threads=args.threads)
    else:
        result = aggregate(root, config)
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()

