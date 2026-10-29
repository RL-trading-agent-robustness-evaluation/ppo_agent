"""Frozen DQN V1 configuration and audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Any
from collections import Counter
import math

import numpy as np
import yaml


@dataclass(frozen=True)
class DQNConfigV1:
    raw: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def toy_gate(self) -> dict[str, Any]:
        return self.raw["toy_gate"]

    @property
    def smoke(self) -> dict[str, Any]:
        return self.raw["smoke"]

    @property
    def memorization(self) -> dict[str, Any]:
        return self.raw["memorization"]

    def validate(self) -> None:
        r = self.raw
        if (r["version"], r["status"], r["algorithm"]) != (1, "frozen", "DQN"):
            raise ValueError("expected frozen DQN V1 config")
        if r["implementation"] != "stable_baselines3" or r["implementation_version"] != "2.9.0":
            raise ValueError("DQN V1 implementation is frozen to stable-baselines3 2.9.0")
        if r["policy"] != "MlpPolicy" or r["device"] != "cpu":
            raise ValueError("DQN V1 uses MlpPolicy on CPU")
        m = r["model"]
        expected_positive = (
            "learning_rate", "gamma", "buffer_size", "learning_starts",
            "batch_size", "train_freq_steps", "gradient_steps",
            "target_update_interval", "tau", "max_grad_norm",
        )
        if any(float(m[key]) <= 0 for key in expected_positive):
            raise ValueError("DQN numeric settings must be positive")
        if not 0 < float(m["gamma"]) <= 1 or not 0 < float(m["tau"]) <= 1:
            raise ValueError("gamma and tau must be in (0, 1]")
        if int(m["learning_starts"]) >= int(m["buffer_size"]):
            raise ValueError("learning_starts must be smaller than buffer_size")
        if int(m["batch_size"]) > int(m["buffer_size"]):
            raise ValueError("batch_size cannot exceed buffer_size")
        if not 0 <= float(m["exploration_final_eps"]) <= float(m["exploration_initial_eps"]) <= 1:
            raise ValueError("invalid epsilon schedule")
        if not 0 < float(m["exploration_fraction"]) <= 1:
            raise ValueError("exploration_fraction must be in (0, 1]")
        e = r["environment"]
        if e["asset"] != "SPY" or e["forbidden_during_development"] != "test":
            raise ValueError("V1 development must use SPY and keep test locked")
        if int(e["episode_length_decisions"]) != 504 or not bool(e["random_start"]):
            raise ValueError("V1 uses random 504-decision train episodes")
        if tuple(float(x) for x in e["action_weights"]) != (0.0, 0.5, 1.0):
            raise ValueError("V1 action weights changed")
        if float(e["transaction_cost_bps"]) != 10.0:
            raise ValueError("V1 cost must remain 10 bps")
        if tuple(int(x) for x in r["formal"]["seeds"]) != (0, 1, 2, 3, 4):
            raise ValueError("formal seed set must remain 0..4")


def load_dqn_config(path: str | Path) -> DQNConfigV1:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = DQNConfigV1(yaml.safe_load(handle))
    config.validate()
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)


def build_sb3_dqn(env: Any, config: DQNConfigV1, *, seed: int):
    """Construct DQN lazily so non-RL tests do not require PyTorch."""

    from stable_baselines3 import DQN
    import torch.nn as nn

    m = config.model
    activations = {"ReLU": nn.ReLU, "Tanh": nn.Tanh}
    if m["activation_fn"] not in activations:
        raise ValueError("unsupported activation_fn")
    return DQN(
        config.raw["policy"],
        env,
        learning_rate=float(m["learning_rate"]),
        buffer_size=int(m["buffer_size"]),
        learning_starts=int(m["learning_starts"]),
        batch_size=int(m["batch_size"]),
        tau=float(m["tau"]),
        gamma=float(m["gamma"]),
        train_freq=(int(m["train_freq_steps"]), "step"),
        gradient_steps=int(m["gradient_steps"]),
        target_update_interval=int(m["target_update_interval"]),
        exploration_fraction=float(m["exploration_fraction"]),
        exploration_initial_eps=float(m["exploration_initial_eps"]),
        exploration_final_eps=float(m["exploration_final_eps"]),
        max_grad_norm=float(m["max_grad_norm"]),
        policy_kwargs={
            "net_arch": [int(x) for x in m["net_arch"]],
            "activation_fn": activations[m["activation_fn"]],
        },
        verbose=0,
        seed=seed,
        device=config.raw["device"],
    )


def parameter_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TrainingAuditCallback:
    """Factory wrapper avoiding a hard SB3 dependency at package import time."""

    @staticmethod
    def create():
        from stable_baselines3.common.callbacks import BaseCallback

        class _AuditCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self.losses: list[float] = []

            def _on_step(self) -> bool:
                loss = self.model.logger.name_to_value.get("train/loss")
                if loss is not None and np.isfinite(float(loss)):
                    self.losses.append(float(loss))
                return True

        return _AuditCallback()


def evaluate_deterministic(model: Any, env: Any, *, seed: int = 0) -> dict[str, Any]:
    """Audit one complete deterministic episode without mutating training state."""

    import torch

    # Evaluation must never leave the network in a training state (dropout /
    # batch-norm updates). DQN V1 uses neither, but assert the mode explicitly.
    model.policy.set_training_mode(False)
    before_parameters = parameter_sha256(model)
    before_replay = int(model.replay_buffer.size())
    observation, reset_info = env.reset(seed=seed)
    rewards: list[float] = []
    nav: list[float] = []
    q_min, q_max = math.inf, -math.inf
    actions: Counter[str] = Counter()
    trajectory: list[dict[str, Any]] = []
    first_nav: float | None = None
    last_nav: float | None = None
    while True:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=model.device).reshape(1, -1)
            q_values = model.q_net(tensor).detach().cpu().numpy()[0]
        if not np.isfinite(q_values).all():
            raise AssertionError("non-finite Q values during deterministic evaluation")
        q_min = min(q_min, float(q_values.min()))
        q_max = max(q_max, float(q_values.max()))
        action, _ = model.predict(observation, deterministic=True)
        action_int = int(np.asarray(action).item())
        next_observation, reward, terminated, truncated, info = env.step(action_int)
        if first_nav is None:
            first_nav = float(info["start_pretrade_nav"])
            nav.append(first_nav)
        last_nav = float(info["end_pretrade_nav"])
        nav.append(last_nav)
        rewards.append(float(reward))
        actions[str(action_int)] += 1
        observation_sha = hashlib.sha256(
            np.ascontiguousarray(observation, dtype=np.float64).tobytes()
        ).hexdigest()
        trajectory.append(
            {
                "decision_date": str(info["decision_date"].date()),
                "execution_date": str(info["execution_date"].date()),
                "valuation_date": str(info["valuation_date"].date()),
                "observation_sha256": observation_sha,
                "requested_action": int(info["requested_action"]),
                "executed_action": info.get("executed_action", action_int),
                "action": action_int,
                **({
                    "order_placed": bool(info["order_placed"]),
                    "order_rejection_reason": info["order_rejection_reason"],
                    "trade_decision_rule": info["trade_decision_rule"],
                } if "trade_decision_rule" in info else {}),
                "target_weight": float(info["target_weight"]),
                "realized_weight": float(info["realized_weight"]),
                "position_before_shares": int(info["position_before_shares"]),
                "position_after_shares": int(info["position_after_shares"]),
                "true_execution_open": float(info["true_execution_open"]),
                "true_valuation_open": float(info["true_valuation_open"]),
                "transaction_cost": float(info["transaction_cost"]),
                "trade_notional": float(info["trade"]["trade_notional"]),
                "turnover": float(info["turnover"]),
                "gross_return": float(info["gross_return"]),
                "net_return": float(info["net_return"]),
                "cash_benchmark_return": float(info["cash_benchmark_return"]),
                "reward": float(reward),
                "start_nav": float(info["start_pretrade_nav"]),
                "end_nav": last_nav,
                "nav": last_nav,
                **{f"q_{i}": float(q_values[i]) for i in range(len(q_values))},
                "epsilon": 0.0,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        observation = next_observation
        if terminated or truncated:
            break
    assert first_nav is not None and last_nav is not None
    reward_sum = float(np.sum(rewards))
    nav_log_change = float(np.log(last_nav / first_nav))
    after_parameters = parameter_sha256(model)
    after_replay = int(model.replay_buffer.size())
    return {
        "reset_info": reset_info,
        "steps": len(rewards),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "initial_nav": first_nav,
        "final_nav": last_nav,
        "final_wealth": last_nav / first_nav,
        "reward_sum": reward_sum,
        "reward_nav_invariant_error": reward_sum - nav_log_change,
        "q_value_min": q_min,
        "q_value_max": q_max,
        "q_values_finite": bool(np.isfinite([q_min, q_max]).all()),
        "action_counts": dict(sorted(actions.items())),
        "action_distribution": {
            action: count / len(rewards) for action, count in sorted(actions.items())
        },
        "parameters_unchanged": before_parameters == after_parameters,
        "replay_size_before": before_replay,
        "replay_size_after": after_replay,
        "replay_unchanged": before_replay == after_replay,
        "nav_finite_positive": bool(np.isfinite(nav).all() and (np.asarray(nav) > 0).all()),
        "trajectory": trajectory,
    }

