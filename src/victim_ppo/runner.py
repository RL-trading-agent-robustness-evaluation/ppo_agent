from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from victimagent.evaluation import optimizer_sha256, scaler_sha256
from victimagent.v4.environment import SingleAssetTradingEnvV4
from victimagent.v4.evaluation import rollout

from .config import PPOConfigV4


def make_environment(bundle, config: PPOConfigV4, *, training: bool):
    shared = config.raw["shared_contract"]
    return SingleAssetTradingEnvV4(
        bundle.market_true,
        bundle.market_obs,
        bundle.observation_builder,
        initial_cash=float(shared["initial_cash"]),
        transaction_cost_bps=float(shared["transaction_cost_bps"]),
        episode_length=int(shared["episode_length_decisions"]) if training else None,
        random_start=bool(shared["random_start"]) if training else False,
        dividend_events=bundle.dividend_events,
        split_events=bundle.split_events,
        action_weights=tuple(float(x) for x in shared["action_weights"]),
    )


def parameter_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def scaler_payload(builder) -> dict[str, Any]:
    state = builder.scaler_state()
    columns = list(builder.scaler.columns)
    return {
        "columns": columns,
        "mean": {column: float(state["mean"][column]) for column in columns},
        "scale": {column: float(state["scale"][column]) for column in columns},
    }


def build_model(env, config: PPOConfigV4, *, seed: int, log_dir: Path):
    import torch
    from stable_baselines3 import PPO

    hp = config.model
    activation = {"Tanh": torch.nn.Tanh, "ReLU": torch.nn.ReLU}[hp["activation_fn"]]
    return PPO(
        config.raw["policy"], env,
        learning_rate=float(hp["learning_rate"]), n_steps=int(hp["n_steps"]),
        batch_size=int(hp["batch_size"]), n_epochs=int(hp["n_epochs"]),
        gamma=float(hp["gamma"]), gae_lambda=float(hp["gae_lambda"]),
        clip_range=float(hp["clip_range"]), normalize_advantage=bool(hp["normalize_advantage"]),
        ent_coef=float(hp["ent_coef"]), vf_coef=float(hp["vf_coef"]),
        max_grad_norm=float(hp["max_grad_norm"]),
        policy_kwargs={"net_arch": list(hp["policy_net_arch"]), "activation_fn": activation},
        seed=seed, device=config.raw["device"], verbose=1, tensorboard_log=None,
    )


def run_training(train_bundle, config: PPOConfigV4, *, seed: int, timesteps: int,
                 checkpoint: Path, log_dir: Path):
    random.seed(seed)
    np.random.seed(seed)
    env = make_environment(train_bundle, config, training=True)
    if env.observation_space.shape != (10,) or env.action_space.n != 3:
        raise AssertionError("PPO did not receive the shared 10-D/3-action V4 interface")
    scaler_before = scaler_sha256(train_bundle.observation_builder)
    model = build_model(env, config, seed=seed, log_dir=log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(str(log_dir), ["csv"]))
    model.learn(total_timesteps=timesteps, reset_num_timesteps=True, progress_bar=False)
    if scaler_sha256(train_bundle.observation_builder) != scaler_before:
        raise AssertionError("train-only scaler mutated during PPO training")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    scaler_path = checkpoint.with_suffix(".scaler.json")
    scaler_path.write_text(json.dumps(scaler_payload(train_bundle.observation_builder), indent=2),
                           encoding="utf-8")
    return model, scaler_path


def run_evaluation(checkpoint: Path, validation_bundle, config: PPOConfigV4, *, seed: int):
    from stable_baselines3 import PPO

    env = make_environment(validation_bundle, config, training=False)
    model = PPO.load(checkpoint, env=env, device="cpu")
    model.policy.set_training_mode(False)
    before = (parameter_sha256(model), optimizer_sha256(model),
              scaler_sha256(validation_bundle.observation_builder))

    def choose(observation, _index):
        action, _state = model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())

    rows, metrics, error = rollout(env, choose, seed=seed)
    after = (parameter_sha256(model), optimizer_sha256(model),
             scaler_sha256(validation_bundle.observation_builder))
    if before != after:
        raise AssertionError("PPO evaluation mutated model, optimizer, or scaler state")
    return rows, metrics, error, {
        "parameters_sha256": before[0], "optimizer_sha256": before[1],
        "scaler_sha256": before[2], "evaluation_state_unchanged": True,
    }

