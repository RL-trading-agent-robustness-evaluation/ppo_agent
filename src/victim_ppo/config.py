from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


V4_COMMIT = "5701400076bd3ad4fa80dbc86d5687f480c26021"


@dataclass(frozen=True)
class PPOConfigV4:
    raw: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def formal(self) -> dict[str, Any]:
        return self.raw["formal"]

    def validate(self) -> None:
        r, shared = self.raw, self.raw["shared_contract"]
        if (r["version"], r["algorithm"], r["implementation_version"]) != (1, "PPO", "2.9.0"):
            raise ValueError("expected PPO V4 configuration for stable-baselines3 2.9.0")
        expected = {
            "victimagent_commit": V4_COMMIT,
            "environment": "SingleAssetTradingEnvV4",
            "observation_dimensions": 10,
            "action_weights": [0.0, 0.5, 1.0],
            "initial_cash": 1_000_000.0,
            "transaction_cost_bps": 10.0,
            "episode_length_decisions": 504,
            "random_start": True,
            "known_period_access_during_development": False,
        }
        for key, value in expected.items():
            if shared.get(key) != value:
                raise ValueError(f"shared V4 contract changed: {key}")
        if shared["train"] != ["2010-01-04", "2018-12-31"]:
            raise ValueError("training dates changed")
        if shared["validation"] != ["2019-01-01", "2021-12-31"]:
            raise ValueError("validation dates changed")
        if tuple(r["formal"]["seeds"]) != (0, 1, 2, 3, 4):
            raise ValueError("formal seeds must be 0..4")
        if int(r["formal"]["total_timesteps_per_seed"]) != 200_000:
            raise ValueError("formal budget must be 200000 steps per seed")
        if int(r["model"]["n_steps"]) % int(r["model"]["batch_size"]):
            raise ValueError("PPO rollout size must be divisible by batch size")
        model = r["model"]
        if model.get("learning_rate_schedule") != "linear":
            raise ValueError("PPO V4 requires the preregistered linear learning-rate schedule")
        initial = float(model["learning_rate_initial"])
        final = float(model["learning_rate_final"])
        if not 0 < final < initial:
            raise ValueError("learning rate must descend from a positive initial to positive final value")


def load_config(path: str | Path) -> PPOConfigV4:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = PPOConfigV4(yaml.safe_load(handle))
    config.validate()
    return config

