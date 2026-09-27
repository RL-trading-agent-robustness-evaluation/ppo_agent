from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


V4_COMMIT = "5701400076bd3ad4fa80dbc86d5687f480c26021"


def git_sha(path: Path) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={path.resolve().as_posix()}", "-C", str(path),
         "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "victimagent/src")]
    from victim_ppo.config import load_config
    from victim_ppo.data import load_development_bundles
    from victim_ppo.runner import make_environment
    from victimagent.v4.environment import SingleAssetTradingEnvV4

    config = load_config(root / "configs/ppo_v4.yaml")
    if git_sha(root / "victimagent") != V4_COMMIT:
        raise RuntimeError("wrong victimagent V4 pin")
    gate = json.loads((root / "reports/gate1_report.json").read_text(encoding="utf-8"))
    if gate.get("gate_1_status") != "PASS":
        raise RuntimeError("shared Gate 1 data pipeline has not passed")
    train, validation = load_development_bundles(root, root)
    train_env = make_environment(train, config, training=True)
    validation_env = make_environment(validation, config, training=False)
    for env in (train_env, validation_env):
        if not isinstance(env, SingleAssetTradingEnvV4):
            raise AssertionError("trainer did not construct SingleAssetTradingEnvV4")
        if env.observation_space.shape != (10,) or env.action_space.n != 3:
            raise AssertionError("shared observation/action contract mismatch")
    if validation.market_true["date"].max() >= pd.Timestamp("2022-01-01"):
        raise AssertionError("known period reached development bundle")
    observation, _ = train_env.reset(seed=0)
    if not train_env.observation_space.contains(observation):
        raise AssertionError("initial V4 observation is invalid")
    result = {
        "status": "PASS", "victimagent_commit": V4_COMMIT,
        "environment": type(train_env).__name__, "observation_shape": [10],
        "actions": [0.0, 0.5, 1.0], "train_rows": len(train.market_true),
        "validation_rows": len(validation.market_true), "known_period_accessed": False,
    }
    out = root / "reports/ppo_v4_preflight.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

