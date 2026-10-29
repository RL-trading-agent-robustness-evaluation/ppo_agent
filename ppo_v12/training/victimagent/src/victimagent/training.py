"""Training-mode entry point for the DQN V1 victim.

Training mode is the *only* mode allowed to mutate learnable state: policy
parameters, the optimizer, the target network, and the replay buffer. It samples
random fixed-length episodes from the train partition and never reads
validation or test data. The observation scaler is fitted once (inside the
bundle) before training and is treated as frozen here.

Evaluation lives in ``victimagent.evaluation``; this module deliberately does
not evaluate, select checkpoints, or touch validation/test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from victimagent.dqn import (
    DQNConfigV1,
    TrainingAuditCallback,
    build_sb3_dqn,
    parameter_sha256,
    seed_everything,
)
from victimagent.environment import SingleAssetTradingEnvV1
from victimagent.evaluation import scaler_sha256


@dataclass
class TrainingResult:
    seed: int
    checkpoint_path: str
    total_timesteps: int
    episode_length: int
    parameter_sha_before: str
    parameter_sha_after: str
    parameters_changed: bool
    replay_size: int
    replay_pos: int | None
    scaler_sha: str
    finite_training_loss: bool
    loss_observations: int
    last_training_loss: float | None
    model: Any = field(default=None, repr=False)


def _replay_position(model: Any) -> int | None:
    buffer = getattr(model, "replay_buffer", None)
    pos = getattr(buffer, "pos", None)
    return int(pos) if pos is not None else None


def train_dqn(
    train_bundle,
    config: DQNConfigV1,
    *,
    seed: int,
    output_directory: str | Path,
    total_timesteps: int | None = None,
    episode_length: int | None = None,
) -> TrainingResult:
    """Train one DQN seed on the train partition only.

    ``total_timesteps`` and ``episode_length`` default to the frozen formal
    values in ``configs/dqn_v1.yaml``; they are overridable only to keep unit
    tests fast and never to alter a formal run.
    """

    partition = getattr(train_bundle, "metadata", {}).get("partition")
    if partition != "train":
        raise ValueError(
            f"TRAIN_PARTITION_ONLY: train_dqn requires a train bundle, got {partition!r}"
        )

    env_cfg = config.raw["environment"]
    if episode_length is None:
        episode_length = int(env_cfg["episode_length_decisions"])
    if total_timesteps is None:
        total_timesteps = int(config.raw["formal"]["total_timesteps_per_seed"])

    seed_everything(seed)
    train_env = SingleAssetTradingEnvV1(
        train_bundle.market_true,
        train_bundle.market_obs,
        train_bundle.observation_builder,
        initial_cash=1_000_000.0,
        transaction_cost_bps=float(env_cfg["transaction_cost_bps"]),
        episode_length=int(episode_length),
        random_start=True,
        dividend_events=train_bundle.dividend_events,
        split_events=train_bundle.split_events,
    )

    model = build_sb3_dqn(train_env, config, seed=seed)
    scaler_before = scaler_sha256(train_bundle.observation_builder)
    parameter_sha_before = parameter_sha256(model)

    callback = TrainingAuditCallback.create()
    model.learn(total_timesteps=int(total_timesteps), callback=callback)

    parameter_sha_after = parameter_sha256(model)
    scaler_after = scaler_sha256(train_bundle.observation_builder)
    if scaler_after != scaler_before:
        raise RuntimeError("SCALER_MUTATION: training changed the frozen scaler state")

    checkpoint = Path(output_directory)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)

    finite_loss = bool(callback.losses and np.isfinite(callback.losses).all())
    return TrainingResult(
        seed=int(seed),
        checkpoint_path=str(checkpoint) + ".zip",
        total_timesteps=int(total_timesteps),
        episode_length=int(episode_length),
        parameter_sha_before=parameter_sha_before,
        parameter_sha_after=parameter_sha_after,
        parameters_changed=parameter_sha_before != parameter_sha_after,
        replay_size=int(model.replay_buffer.size()),
        replay_pos=_replay_position(model),
        scaler_sha=scaler_after,
        finite_training_loss=finite_loss,
        loss_observations=len(callback.losses),
        last_training_loss=callback.losses[-1] if callback.losses else None,
        model=model,
    )
