from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPECTED_PIN = "f4988db5c98d9248533c4ca852cc48ca3f6b7aad"
ARMS = ("ppo_v9_gate_nav", "ppo_v9_gate_relative")
FOLDS = ("fold_1", "fold_2", "fold_3")
SEEDS = (0, 1, 2, 3, 4)
REQUESTED_TIMESTEPS = 500_000
EXPECTED_ACTUAL_TIMESTEPS = 501_760
RAW_FILES = (
    "crsp_dsf_spy_1993_2008.csv",
    "crsp_dsf_spy_tlt_gld_2009_2025.csv",
    "stkdistributions_spy_1993_2008.csv",
    "stkdistributions_spy_tlt_gld_2009_2025.csv",
    "DGS3MO_1993_2008.csv",
    "DGS3MO_2009_2025.csv",
)


@dataclass(frozen=True)
class PPOConfigV9:
    path: Path
    raw: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    def upstream_arm(self, arm: str) -> str:
        if arm not in ARMS:
            raise ValueError(f"unknown PPO V9 arm: {arm}")
        return str(self.raw["arms"][arm]["victimagent_arm"])


def load_config(path: str | Path) -> PPOConfigV9:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    shared, formal = raw["shared_contract"], raw["formal"]
    fixed = {
        "version": 1,
        "status": "preregistered_before_ppo_v9_training",
        "algorithm": "PPO",
        "implementation_version": "2.9.0",
    }
    for key, value in fixed.items():
        if raw.get(key) != value:
            raise ValueError(f"PPO V9 fixed contract changed: {key}")
    if shared["victimagent_commit"] != EXPECTED_PIN:
        raise ValueError("unexpected victimagent V9 pin")
    if tuple(shared["folds"]) != FOLDS or tuple(shared["seeds"]) != SEEDS:
        raise ValueError("PPO V9 folds or seeds changed")
    if tuple(raw["arms"]) != ARMS:
        raise ValueError("PPO V9 reward arms changed")
    if formal["requested_timesteps_per_run"] != REQUESTED_TIMESTEPS:
        raise ValueError("formal PPO budget changed")
    if formal["expected_sb3_timesteps_per_run"] != EXPECTED_ACTUAL_TIMESTEPS:
        raise ValueError("documented SB3 rollout budget changed")
    if formal["record_revision"] != "v9spec1":
        raise ValueError("formal record schema revision changed")
    if raw["model"]["n_steps"] != 2048 or raw["model"]["ent_coef"] != 0.0:
        raise ValueError("frozen PPO rollout or entropy setting changed")
    if raw["formal"]["final_arm"] != "ppo_v9_gate_relative":
        raise ValueError("the preregistered final arm changed")
    return PPOConfigV9(path, raw)


def missing_raw_files(victimagent_root: Path) -> list[str]:
    raw = victimagent_root / "data" / "raw"
    return [name for name in RAW_FILES if not (raw / name).is_file()]

