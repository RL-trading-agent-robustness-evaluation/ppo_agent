"""Evaluation-mode entry point for the DQN V1 victim.

Evaluation is read-only with respect to *every* piece of learnable or fitted
state. It loads a fixed checkpoint, acts deterministically over the full fixed
evaluation period, and asserts that parameters, optimizer state, replay buffer,
and the observation scaler are byte-identical before and after. It refuses the
locked test partition unless an explicit one-shot Gate C2 authorization is
passed.

Core parameter/replay immutability and the trajectory are produced by
``victimagent.dqn.evaluate_deterministic``; this module reuses that single
evaluation loop and adds the scaler and optimizer checks around it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from victimagent.dqn import (
    DQNConfigV1,
    evaluate_deterministic,
    parameter_sha256,
)
from victimagent.environment import SingleAssetTradingEnvV1


class LockedPartitionError(RuntimeError):
    """Raised when the locked test partition is accessed without Gate C2."""


@dataclass
class EvaluationResult:
    partition: str
    seed: int
    checkpoint_path: str
    steps: int
    terminated: bool
    truncated: bool
    natural_termination: bool
    final_wealth: float
    reward_nav_invariant_error: float
    q_values_finite: bool
    nav_finite_positive: bool
    action_distribution: dict[str, float]
    total_transaction_cost: float
    total_turnover: float
    parameters_unchanged: bool
    optimizer_unchanged: bool
    replay_unchanged: bool
    scaler_unchanged: bool
    parameter_sha: str
    optimizer_sha: str
    scaler_sha: str
    replay_size: int
    replay_pos: int | None
    trajectory_path: str | None
    test_partition_accessed: bool
    metrics: dict[str, Any] = field(default_factory=dict)


def performance_metrics(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    """Baseline-B0-comparable performance metrics from an eval trajectory.

    Uses the same formulas as ``victimagent.baselines.run_baseline`` so that a
    DQN checkpoint and the scripted baselines can be read in one table:
    annualized excess-return Sharpe against the aligned cash return, CAGR from
    calendar span, drawdown from the NAV path, turnover, cost, and exposure.
    """

    if not trajectory:
        raise ValueError("cannot compute metrics from an empty trajectory")

    net_returns = np.asarray([row["net_return"] for row in trajectory], dtype=float)
    cash_returns = np.asarray(
        [row["cash_benchmark_return"] for row in trajectory], dtype=float
    )
    turnovers = np.asarray([row["turnover"] for row in trajectory], dtype=float)
    exposures = np.asarray([row["realized_weight"] for row in trajectory], dtype=float)
    nav_path = np.asarray(
        [trajectory[0]["start_nav"]] + [row["end_nav"] for row in trajectory],
        dtype=float,
    )
    first_nav = float(nav_path[0])
    last_nav = float(nav_path[-1])

    first_execution = pd.Timestamp(trajectory[0]["execution_date"])
    last_valuation = pd.Timestamp(trajectory[-1]["valuation_date"])
    elapsed_years = (last_valuation - first_execution).days / 365.25
    if elapsed_years <= 0:
        raise AssertionError("evaluation period must span positive calendar time")

    excess = net_returns - cash_returns
    excess_std = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    sharpe = (
        float(np.mean(excess) / excess_std * np.sqrt(252.0))
        if excess_std > 1e-12
        else None
    )
    drawdown = nav_path / np.maximum.accumulate(nav_path) - 1.0
    number_of_trades = int(
        sum(1 for row in trajectory if abs(float(row["trade_notional"])) > 1e-8)
    )
    # The policy's emitted action exists even when a V3 execution rule holds shares.
    # For V1/V2 the emitted and executed action IDs are identical.
    actions: Counter[str] = Counter(str(int(row["action"])) for row in trajectory)
    steps = len(trajectory)

    return {
        "steps": steps,
        "start_date": str(first_execution.date()),
        "end_date": str(last_valuation.date()),
        "final_wealth": last_nav / first_nav,
        "total_return": last_nav / first_nav - 1.0,
        "cagr": (last_nav / first_nav) ** (1.0 / elapsed_years) - 1.0,
        "annualized_mean_excess_return": float(np.mean(excess)) * 252.0,
        "sharpe_excess_cash": sharpe,
        "annualized_volatility": (
            float(np.std(net_returns, ddof=1) * np.sqrt(252.0)) if steps > 1 else 0.0
        ),
        "max_drawdown": float(np.min(drawdown)),
        "average_exposure": float(np.mean(exposures)),
        "total_turnover": float(np.sum(turnovers)),
        "number_of_trades": number_of_trades,
        "total_transaction_cost": float(
            sum(row["transaction_cost"] for row in trajectory)
        ),
        "action_distribution": {
            label: count / steps for label, count in sorted(actions.items())
        },
    }


def scaler_fingerprint(observation_builder) -> dict[str, Any]:
    """Ordered, JSON-serializable view of the frozen train-fit scaler."""

    state = observation_builder.scaler_state()
    columns = tuple(observation_builder.scaler.columns)
    return {
        "columns": list(columns),
        "mean": [float(state["mean"][c]) for c in columns],
        "scale": [float(state["scale"][c]) for c in columns],
    }


def scaler_sha256(observation_builder) -> str:
    payload = json.dumps(scaler_fingerprint(observation_builder), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def optimizer_sha256(model: Any) -> str:
    import torch

    optimizer = model.policy.optimizer
    state = optimizer.state_dict()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(state.get("param_groups", []), sort_keys=True, default=str).encode(
            "utf-8"
        )
    )
    per_param = state.get("state", {})
    for key in sorted(per_param, key=str):
        entry = per_param[key]
        for name in sorted(entry, key=str):
            value = entry[name]
            digest.update(f"{key}:{name}".encode("utf-8"))
            if torch.is_tensor(value):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def _replay_position(model: Any) -> int | None:
    buffer = getattr(model, "replay_buffer", None)
    pos = getattr(buffer, "pos", None)
    return int(pos) if pos is not None else None


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    evaluation_bundle,
    config: DQNConfigV1,
    *,
    seed: int,
    partition: str,
    allow_test: bool = False,
    gate: str | None = None,
    trajectory_dir: str | Path | None = "artifacts/trajectories",
) -> EvaluationResult:
    """Deterministically evaluate a checkpoint without mutating any state.

    ``partition == "test"`` is refused unless ``allow_test=True`` and
    ``gate == "C2"`` are both supplied; that authorization is the one-shot Gate
    C2 escape hatch and must never be the default.
    """

    if partition == "test" and not (allow_test and gate == "C2"):
        raise LockedPartitionError("Test partition is locked until Gate C2.")

    bundle_partition = getattr(evaluation_bundle, "metadata", {}).get("partition")
    if bundle_partition != partition:
        raise ValueError(
            "EVALUATION_PARTITION_MISMATCH: "
            f"bundle is {bundle_partition!r}, requested {partition!r}"
        )

    from stable_baselines3 import DQN

    env_cfg = config.raw["environment"]
    eval_env = SingleAssetTradingEnvV1(
        evaluation_bundle.market_true,
        evaluation_bundle.market_obs,
        evaluation_bundle.observation_builder,
        initial_cash=1_000_000.0,
        transaction_cost_bps=float(env_cfg["transaction_cost_bps"]),
        episode_length=None,
        random_start=False,
        dividend_events=evaluation_bundle.dividend_events,
        split_events=evaluation_bundle.split_events,
    )

    model = DQN.load(Path(checkpoint_path), env=eval_env, device="cpu")

    scaler_before = scaler_sha256(evaluation_bundle.observation_builder)
    optimizer_before = optimizer_sha256(model)
    parameter_before = parameter_sha256(model)
    replay_before = int(model.replay_buffer.size())
    replay_pos_before = _replay_position(model)

    audit = evaluate_deterministic(model, eval_env, seed=seed)

    scaler_after = scaler_sha256(evaluation_bundle.observation_builder)
    optimizer_after = optimizer_sha256(model)
    parameter_after = parameter_sha256(model)
    replay_after = int(model.replay_buffer.size())
    replay_pos_after = _replay_position(model)

    trajectory = audit.pop("trajectory")
    trajectory_path: str | None = None
    if trajectory_dir is not None:
        out_dir = Path(trajectory_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"dqn_v1_seed{seed}_{partition}.csv"
        pd.DataFrame(trajectory).to_csv(out_path, index=False)
        trajectory_path = str(out_path)

    metrics = performance_metrics(trajectory)
    total_cost = metrics["total_transaction_cost"]
    total_turnover = metrics["total_turnover"]

    return EvaluationResult(
        partition=partition,
        seed=int(seed),
        checkpoint_path=str(checkpoint_path),
        steps=int(audit["steps"]),
        terminated=bool(audit["terminated"]),
        truncated=bool(audit["truncated"]),
        natural_termination=bool(audit["terminated"] and not audit["truncated"]),
        final_wealth=float(audit["final_wealth"]),
        reward_nav_invariant_error=float(audit["reward_nav_invariant_error"]),
        q_values_finite=bool(audit["q_values_finite"]),
        nav_finite_positive=bool(audit["nav_finite_positive"]),
        action_distribution=audit["action_distribution"],
        total_transaction_cost=total_cost,
        total_turnover=total_turnover,
        parameters_unchanged=parameter_before == parameter_after,
        optimizer_unchanged=optimizer_before == optimizer_after,
        replay_unchanged=(replay_before == replay_after)
        and (replay_pos_before == replay_pos_after),
        scaler_unchanged=scaler_before == scaler_after,
        parameter_sha=parameter_after,
        optimizer_sha=optimizer_after,
        scaler_sha=scaler_after,
        replay_size=replay_after,
        replay_pos=replay_pos_after,
        trajectory_path=trajectory_path,
        test_partition_accessed=(partition == "test"),
        metrics=metrics,
    )
