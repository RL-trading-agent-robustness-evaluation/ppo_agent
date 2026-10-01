from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from victim_ppo.v9.config import (EXPECTED_ACTUAL_TIMESTEPS, EXPECTED_PIN, FOLDS,
                                  RAW_FILES, REQUESTED_TIMESTEPS, SEEDS,
                                  missing_raw_files)

ARMS = ("ppo_v10_gate_nav", "ppo_v10_gate_nav_streak")
BETA = 0.10
STREAK_CAP = 5


@dataclass(frozen=True)
class PPOConfigV10:
    path: Path
    raw: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    def uses_streak_reward(self, arm: str) -> bool:
        if arm not in ARMS:
            raise ValueError(f"unknown PPO V10 arm: {arm}")
        return bool(self.raw["arms"][arm]["streak_reward"])


def load_config(path: str | Path) -> PPOConfigV10:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    shared, formal, reward = raw["shared_contract"], raw["formal"], raw["streak_reward"]
    fixed = {
        "version": 1,
        "status": "frozen_before_ppo_v10_training",
        "algorithm": "PPO",
        "implementation_version": "2.9.0",
    }
    for key, value in fixed.items():
        if raw.get(key) != value:
            raise ValueError(f"PPO V10 fixed contract changed: {key}")
    if shared["victimagent_commit"] != EXPECTED_PIN:
        raise ValueError("unexpected victimagent V9 parent pin")
    if tuple(shared["folds"]) != FOLDS or tuple(shared["seeds"]) != SEEDS:
        raise ValueError("PPO V10 folds or seeds changed")
    if tuple(raw["arms"]) != ARMS:
        raise ValueError("PPO V10 arms changed")
    if float(reward["beta"]) != BETA or int(reward["streak_cap"]) != STREAK_CAP:
        raise ValueError("PPO V10 streak reward parameters changed")
    if float(reward["positive_threshold"]) != 0.0 or not reward["zero_breaks_streak"]:
        raise ValueError("PPO V10 positive-return definition changed")
    if formal["requested_timesteps_per_run"] != REQUESTED_TIMESTEPS:
        raise ValueError("formal PPO budget changed")
    if formal["expected_sb3_timesteps_per_run"] != EXPECTED_ACTUAL_TIMESTEPS:
        raise ValueError("documented SB3 rollout budget changed")
    if formal["record_revision"] != "v10spec1":
        raise ValueError("formal record schema revision changed")
    if formal["paired_control"] != ARMS[0] or formal["paired_treatment"] != ARMS[1]:
        raise ValueError("paired V10 contrast changed")
    if raw["model"]["n_steps"] != 2048 or raw["model"]["ent_coef"] != 0.0:
        raise ValueError("frozen PPO rollout or entropy setting changed")
    if raw["known_period"]["enabled_by_this_runner"] is not False:
        raise ValueError("V10 development runner must not enable known-period access")
    return PPOConfigV10(path, raw)


__all__ = [
    "ARMS", "BETA", "STREAK_CAP", "EXPECTED_ACTUAL_TIMESTEPS", "EXPECTED_PIN",
    "FOLDS", "RAW_FILES", "REQUESTED_TIMESTEPS", "SEEDS", "PPOConfigV10",
    "load_config", "missing_raw_files",
]

