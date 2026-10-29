"""Frozen victim-training episode protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingProtocol:
    primary_asset: str
    episode_length: int
    random_start: bool
    initial_cash: float
    force_liquidation_at_end: bool
    transaction_cost_bps: float
    action_weights: tuple[float, ...]
    training_seeds: tuple[int, ...]
    normalization_fit_partition: str
    update_normalization_during_training: bool
    update_normalization_during_evaluation: bool
    test_locked: bool
    pre_test_gate: str
    one_shot_test_gate: str

    def validate(self) -> None:
        if self.primary_asset != "SPY":
            raise ValueError("V1 primary asset must remain SPY")
        if self.episode_length != 504:
            raise ValueError("V1 training episodes are frozen at 504 decisions")
        if not self.random_start:
            raise ValueError("V1 training must use random train-only starts")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.force_liquidation_at_end:
            raise ValueError("V1 must not charge a synthetic terminal liquidation")
        if self.transaction_cost_bps != 10.0:
            raise ValueError("V1 primary transaction cost is frozen at 10 bps")
        if self.action_weights != (0.0, 0.5, 1.0):
            raise ValueError("V1 action weights must be 0%, 50%, and 100%")
        if len(self.training_seeds) < 5 or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ValueError("at least five unique training seeds are required")
        if self.normalization_fit_partition != "train":
            raise ValueError("normalization must be fitted on train only")
        if self.update_normalization_during_training:
            raise ValueError("normalization is frozen before policy training")
        if self.update_normalization_during_evaluation:
            raise ValueError("evaluation must not update normalization")
        if not self.test_locked:
            raise ValueError("test must remain locked during model development")
        if (self.pre_test_gate, self.one_shot_test_gate) != ("C1", "C2"):
            raise ValueError("clean evaluation gates must remain C1 then C2")


def load_training_protocol(path: str | Path) -> TrainingProtocol:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    protocol = TrainingProtocol(
        primary_asset=str(raw["primary_asset"]),
        episode_length=int(raw["episode"]["length_decisions"]),
        random_start=bool(raw["episode"]["random_start"]),
        initial_cash=float(raw["episode"]["initial_cash"]),
        force_liquidation_at_end=bool(raw["episode"]["force_liquidation_at_end"]),
        transaction_cost_bps=float(raw["execution"]["transaction_cost_bps"]),
        action_weights=tuple(float(x) for x in raw["execution"]["action_weights"]),
        training_seeds=tuple(int(x) for x in raw["reproducibility"]["training_seeds"]),
        normalization_fit_partition=str(
            raw["preprocessing"]["normalization_fit_partition"]
        ),
        update_normalization_during_training=bool(
            raw["preprocessing"]["update_during_training"]
        ),
        update_normalization_during_evaluation=bool(
            raw["preprocessing"]["update_during_evaluation"]
        ),
        test_locked=bool(raw["evaluation"]["test_locked_until_protocol_freeze"]),
        pre_test_gate=str(raw["evaluation"]["pre_test_gate"]),
        one_shot_test_gate=str(raw["evaluation"]["one_shot_test_gate"]),
    )
    protocol.validate()
    return protocol
