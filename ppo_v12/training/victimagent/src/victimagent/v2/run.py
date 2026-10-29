"""Run one V2 candidate across the rolling folds and append its registry record.

Training uses the shared, frozen ledger/reward/execution engine
(``SingleAssetTradingEnvV1``) with only the candidate's action representation
varied. Every fold fits its own train-only scaler (via ``prepare_asset_bundle``
on a one-fold contract) and evaluates deterministically on that fold's
validation window. The known-period window is never opened.

Technical eligibility is applied without looking at performance, per the
contract; failed fold-seeds are retained, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from victimagent.data.real_adapter import prepare_asset_bundle
from victimagent.dqn import (
    TrainingAuditCallback,
    evaluate_deterministic,
    parameter_sha256,
    seed_everything,
)
from victimagent.environment import SingleAssetTradingEnvV1
from victimagent.features import MARKET_FEATURES
from victimagent.evaluation import (
    optimizer_sha256,
    performance_metrics,
    scaler_sha256,
)
from victimagent.v2.folds import EPISODE_DECISIONS, audit_folds, load_rolling_folds
from victimagent.v2.registry import load_records, sha256_file
from victimagent.v2.search import (
    CandidateSpec,
    budget_and_seeds,
    load_v2_search,
    resolve_candidate,
    selection_requirements,
)
from victimagent.v2.select import select_stage_winner

REWARD_NAV_TOLERANCE = 1.0e-10


def _development_only(
    market: pd.DataFrame, events: pd.DataFrame, forbidden_start: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Physically remove the known period before any V2 audit or bundle build."""

    cutoff = pd.Timestamp(forbidden_start)
    market_dates = pd.to_datetime(market["date"], errors="raise")
    isolated_market = market.loc[market_dates < cutoff].copy()
    if not isolated_market.empty and pd.to_datetime(isolated_market["date"]).max() >= cutoff:
        raise AssertionError("known-period market rows survived hard isolation")

    event_date_col = next(
        (name for name in ("exdt", "disexdt") if name in events.columns),
        None,
    )
    if event_date_col is None:
        raise KeyError("distribution events need exdt or disexdt for hard isolation")
    event_dates = pd.to_datetime(events[event_date_col], errors="raise")
    isolated_events = events.loc[event_dates < cutoff].copy()
    return isolated_market, isolated_events


def _responsiveness_diagnostics(model, env, *, seed: int, context_width: int) -> dict[str, Any]:
    """Causal offline policy diagnostic over validation observations.

    Market context is cyclically permuted and then neutralized while account
    state stays paired with its original row.  This does not create a P&L path.
    """

    import torch

    observations: list[np.ndarray] = []
    observation, _ = env.reset(seed=seed)
    while True:
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(int(np.asarray(action).item()))
        if terminated or truncated:
            break
    matrix = np.asarray(observations, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) < 2 or not 0 < context_width < matrix.shape[1]:
        raise ValueError("invalid observation matrix for responsiveness audit")

    def q_values(values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(values, dtype=torch.float32, device=model.device)
            result = model.q_net(tensor).detach().cpu().numpy()
        if result.ndim != 2 or not np.isfinite(result).all():
            raise AssertionError("responsiveness audit requires finite Q values")
        return result

    original_q = q_values(matrix)
    original_actions = np.argmax(original_q, axis=1)
    cyclic = matrix.copy()
    cyclic[:, :context_width] = np.roll(
        matrix[:, :context_width], max(1, len(matrix) // 2), axis=0
    )
    neutral = matrix.copy()
    neutral[:, :context_width] = 0.0
    cyclic_actions = np.argmax(q_values(cyclic), axis=1)
    neutral_actions = np.argmax(q_values(neutral), axis=1)

    unique_actions = int(len(np.unique(original_actions)))
    q_range_max = float(np.max(np.ptp(original_q, axis=0)))
    cyclic_changes = int(np.count_nonzero(original_actions != cyclic_actions))
    neutral_changes = int(np.count_nonzero(original_actions != neutral_actions))
    q_varies = bool(q_range_max > 0.0)
    market_sensitive = bool(cyclic_changes > 0 or neutral_changes > 0)
    eligible = bool(unique_actions >= 2 and q_varies and market_sensitive)
    return {
        "steps": int(len(matrix)),
        "unique_actions": unique_actions,
        "q_values_vary_across_states": q_varies,
        "q_range_max": q_range_max,
        "cyclic_context_action_changes": cyclic_changes,
        "cyclic_context_disagreement_rate": cyclic_changes / len(matrix),
        "neutral_context_action_changes": neutral_changes,
        "neutral_context_disagreement_rate": neutral_changes / len(matrix),
        "market_context_changes_actions": market_sensitive,
        "eligible": eligible,
    }


def build_v2_dqn(env, hyperparameters: dict[str, Any], *, seed: int):
    from stable_baselines3 import DQN
    import torch.nn as nn

    activations = {"ReLU": nn.ReLU, "Tanh": nn.Tanh}
    hp = hyperparameters
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=float(hp["learning_rate"]),
        buffer_size=int(hp["buffer_size"]),
        learning_starts=int(hp["learning_starts"]),
        batch_size=int(hp["batch_size"]),
        tau=float(hp["tau"]),
        gamma=float(hp["gamma"]),
        train_freq=(int(hp["train_freq_steps"]), "step"),
        gradient_steps=int(hp["gradient_steps"]),
        target_update_interval=int(hp["target_update_interval"]),
        exploration_fraction=float(hp["exploration_fraction"]),
        exploration_initial_eps=float(hp.get("exploration_initial_eps", 1.0)),
        exploration_final_eps=float(hp["exploration_final_eps"]),
        max_grad_norm=float(hp["max_grad_norm"]),
        policy_kwargs={
            "net_arch": [int(x) for x in hp["net_arch"]],
            "activation_fn": activations[hp.get("activation_fn", "ReLU")],
        },
        verbose=0,
        seed=int(seed),
        device="cpu",
    )


def _env(bundle, spec: CandidateSpec, *, episode_length, random_start):
    return SingleAssetTradingEnvV1(
        bundle.market_true,
        bundle.market_obs,
        bundle.observation_builder,
        initial_cash=1_000_000.0,
        transaction_cost_bps=10.0,
        episode_length=episode_length,
        random_start=random_start,
        dividend_events=bundle.dividend_events,
        split_events=bundle.split_events,
        action_weights=spec.action_weights,
    )


@dataclass
class FoldSeedResult:
    fold_id: str
    seed: int
    checkpoint_path: str
    trajectory_path: str
    metrics: dict[str, Any]
    technical_gates: dict[str, bool]
    eligible: bool


def run_fold_seed(
    *,
    root: Path,
    spec: CandidateSpec,
    train_bundle,
    val_bundle,
    fold_id: str,
    seed: int,
    timesteps: int,
    episode_length: int,
    artifact_id: str | None = None,
) -> FoldSeedResult:
    from stable_baselines3 import DQN

    # Finalists retrain a source spec, but their artifacts must have the
    # finalist record ID; otherwise they overwrite the screening evidence.
    run_id = artifact_id or spec.experiment_id
    ckpt_dir = root / "artifacts" / "checkpoints" / "v2"
    traj_dir = root / "artifacts" / "trajectories" / "v2"
    checkpoint = ckpt_dir / f"{run_id}_{fold_id}_seed{seed}"
    traj_path = traj_dir / f"{run_id}_{fold_id}_seed{seed}_validation.csv"
    if artifact_id is not None and (Path(str(checkpoint) + ".zip").exists() or traj_path.exists()):
        raise FileExistsError(f"existing finalist artifacts for {run_id}/{fold_id}/seed{seed}; do not overwrite")

    seed_everything(seed)
    train_env = _env(train_bundle, spec, episode_length=episode_length, random_start=True)
    model = build_v2_dqn(train_env, spec.hyperparameters, seed=seed)

    scaler_before = scaler_sha256(train_bundle.observation_builder)
    param_before = parameter_sha256(model)
    callback = TrainingAuditCallback.create()
    model.learn(total_timesteps=int(timesteps), callback=callback)
    param_after_train = parameter_sha256(model)
    if scaler_sha256(train_bundle.observation_builder) != scaler_before:
        raise RuntimeError("SCALER_MUTATION: training changed the frozen scaler state")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)

    # Evaluation on this fold's validation window: read-only in every sense.
    val_env = _env(val_bundle, spec, episode_length=None, random_start=False)
    eval_model = DQN.load(checkpoint, env=val_env, device="cpu")
    scaler_eval_before = scaler_sha256(val_bundle.observation_builder)
    optimizer_before = optimizer_sha256(eval_model)
    param_eval_before = parameter_sha256(eval_model)
    replay_before = int(eval_model.replay_buffer.size())

    audit = evaluate_deterministic(eval_model, val_env, seed=seed)
    responsiveness = _responsiveness_diagnostics(
        eval_model,
        val_env,
        seed=seed,
        context_width=len(val_bundle.observation_builder.context_features),
    )

    scaler_eval_after = scaler_sha256(val_bundle.observation_builder)
    optimizer_after = optimizer_sha256(eval_model)
    param_eval_after = parameter_sha256(eval_model)
    replay_after = int(eval_model.replay_buffer.size())

    trajectory = audit.pop("trajectory")
    traj_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trajectory).to_csv(traj_path, index=False)

    metrics = performance_metrics(trajectory)
    metrics["responsiveness"] = responsiveness

    gates = {
        "finite_training_loss": bool(callback.losses)
        and bool(np.isfinite(callback.losses).all()),
        "training_updated_parameters": param_before != param_after_train,
        "reward_nav_invariant": abs(float(audit["reward_nav_invariant_error"]))
        <= REWARD_NAV_TOLERANCE,
        "evaluation_parameters_unchanged": param_eval_before == param_eval_after,
        "evaluation_optimizer_unchanged": optimizer_before == optimizer_after,
        "evaluation_replay_unchanged": replay_before == replay_after,
        "evaluation_scaler_unchanged": scaler_eval_before == scaler_eval_after,
        "natural_termination": bool(audit["terminated"]) and not bool(audit["truncated"]),
        "nav_finite_positive": bool(audit["nav_finite_positive"]),
        "q_values_finite": bool(audit["q_values_finite"]),
        "state_responsive": bool(responsiveness["eligible"]),
    }
    eligible = all(gates.values())
    return FoldSeedResult(
        fold_id=fold_id,
        seed=int(seed),
        checkpoint_path=str(checkpoint.relative_to(root)) + ".zip",
        trajectory_path=str(traj_path.relative_to(root)),
        metrics=metrics,
        technical_gates=gates,
        eligible=eligible,
    )


def _git_commit(root: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def _median(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.median(clean)) if clean else None


def _train_spec_over_folds(
    *,
    root: Path,
    spec: CandidateSpec,
    market: pd.DataFrame,
    events: pd.DataFrame,
    fold_contract,
    folds,
    seeds: list[int],
    timesteps: int,
    artifact_id: str | None = None,
) -> list[FoldSeedResult]:
    """Train+evaluate one resolved config across folds/seeds (shared by the
    screening and finalist runners). The action set and observation layout come
    entirely from ``spec``; the ledger/reward/execution engine is unchanged."""

    results: list[FoldSeedResult] = []
    for fold in folds:
        contract = fold_contract.split_contract_for(fold)
        train_bundle = prepare_asset_bundle(
            market, events, contract, ticker="SPY", partition="train",
            market_features=spec.market_features, include_cash_rate=spec.include_cash_rate,
        )
        val_bundle = prepare_asset_bundle(
            market, events, contract, ticker="SPY", partition="validation",
            market_features=spec.market_features, include_cash_rate=spec.include_cash_rate,
        )
        for seed in seeds:
            results.append(
                run_fold_seed(
                    root=root,
                    spec=spec,
                    train_bundle=train_bundle,
                    val_bundle=val_bundle,
                    fold_id=fold.fold_id,
                    seed=seed,
                    timesteps=timesteps,
                    episode_length=EPISODE_DECISIONS,
                    artifact_id=artifact_id,
                )
            )
    return results


def _score_results(
    results: list[FoldSeedResult],
) -> tuple[list[dict[str, Any]], list[float], bool]:
    per_fold_seed = [
        {
            "fold": r.fold_id,
            "seed": r.seed,
            "eligible": r.eligible,
            "sharpe_excess_cash": r.metrics["sharpe_excess_cash"],
            "final_wealth": r.metrics["final_wealth"],
            "annualized_mean_excess_return": r.metrics["annualized_mean_excess_return"],
            "max_drawdown": r.metrics["max_drawdown"],
            "average_exposure": r.metrics["average_exposure"],
            "total_turnover": r.metrics["total_turnover"],
            "total_transaction_cost": r.metrics["total_transaction_cost"],
            "action_distribution": r.metrics["action_distribution"],
            "technical_gates": r.technical_gates,
            "trajectory_path": r.trajectory_path,
        }
        for r in results
    ]
    eligible_sharpes = [
        row["sharpe_excess_cash"] for row in per_fold_seed if row["eligible"]
    ]
    all_eligible = bool(results) and all(r.eligible for r in results)
    return per_fold_seed, eligible_sharpes, all_eligible


def _build_record(
    *,
    experiment_id: str,
    stage: str,
    parent_experiment_id: str | None,
    hypothesis: str,
    changed_category: str,
    changed_fields: dict[str, Any],
    root: Path,
    search_path: Path,
    market_path: Path,
    event_path: Path,
    seeds: list[int],
    timesteps: int,
    folds,
    results: list[FoldSeedResult],
) -> dict[str, Any]:
    per_fold_seed, eligible_sharpes, all_eligible = _score_results(results)
    return {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "parent_experiment_id": parent_experiment_id,
        "hypothesis": hypothesis.strip(),
        "changed_category": changed_category,
        "changed_fields": changed_fields,
        "git_commit": _git_commit(root),
        "config_path": "configs/dqn_v2_search.yaml",
        "config_sha256": sha256_file(search_path),
        "data_sha256": {
            "processed_market": sha256_file(market_path),
            "distribution_events": sha256_file(event_path),
        },
        "split_sha256": sha256_file(root / "configs" / "v2_rolling_folds.yaml"),
        "seeds": sorted(seeds),
        "training_budget": {"timesteps_per_fold_seed": int(timesteps), "cpu_hours": None},
        "folds": [f.fold_id for f in folds],
        "status": "pass" if all_eligible else "fail",
        "technical_gates": {
            "all_fold_seeds_eligible": all_eligible,
            "fold_seed_count": len(results),
        },
        "validation_metrics": {
            "primary_metric": "median_sharpe_excess_lagged_cash",
            "median_sharpe_excess_lagged_cash": _median(eligible_sharpes),
            "per_fold_seed": per_fold_seed,
        },
        "trajectory_paths": [r.trajectory_path for r in results],
        "decision": "pending",
        "decision_reason": None,
        "known_period_accessed": False,
    }


def run_candidate(
    *,
    root: Path,
    data_root: Path,
    stage: str,
    candidate_id: str,
    hypothesis: str,
    mode: str = "screening",
    folds_subset: list[str] | None = None,
    seeds_subset: list[int] | None = None,
    timesteps_override: int | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Train+evaluate a candidate across folds/seeds and return its record.

    The record still needs appending by the caller (``append_record``); returning
    it first keeps the pure computation testable without touching the registry.

    ``observation_ablation`` and ``dqn_hyperparameters`` candidates inherit the
    winning configuration of the prior stage, resolved from the append-only
    registry (``registry_path``, default ``experiments/v2_registry.jsonl``).
    """

    if not hypothesis.strip():
        raise ValueError("a non-empty hypothesis is required before running a candidate")

    fold_contract = load_rolling_folds(root / "configs" / "v2_rolling_folds.yaml")
    search_path = root / "configs" / "dqn_v2_search.yaml"
    search = load_v2_search(search_path)
    if registry_path is None:
        registry_path = root / "experiments" / "v2_registry.jsonl"
    records = load_records(registry_path) if Path(registry_path).exists() else []
    spec = resolve_candidate(search, stage, candidate_id, records=records)

    market_path = data_root / "data" / "processed" / "market_base_with_splits.csv"
    event_path = data_root / "data" / "interim" / "distribution_events.csv"
    market = pd.read_csv(market_path)
    events = pd.read_csv(event_path)
    market, events = _development_only(
        market,
        events,
        search["immutable_environment"]["forbidden_selection_period_start"],
    )

    audit = audit_folds(fold_contract, market, events, ticker="SPY")
    if audit.status != "PASS":
        raise RuntimeError(f"fold coverage audit must PASS before training: {audit.to_dict()}")

    seeds, timesteps = budget_and_seeds(search, mode)
    if seeds_subset is not None:
        seeds = [s for s in seeds if s in set(seeds_subset)]
    if timesteps_override is not None:
        timesteps = int(timesteps_override)
    folds = [f for f in fold_contract.folds if folds_subset is None or f.fold_id in set(folds_subset)]

    results = _train_spec_over_folds(
        root=root, spec=spec, market=market, events=events,
        fold_contract=fold_contract, folds=folds, seeds=seeds, timesteps=timesteps,
    )
    return _build_record(
        experiment_id=spec.experiment_id,
        stage=spec.stage,
        parent_experiment_id=spec.parent_experiment_id,
        hypothesis=hypothesis,
        changed_category=spec.changed_category,
        changed_fields=spec.changed_fields,
        root=root, search_path=search_path, market_path=market_path, event_path=event_path,
        seeds=seeds, timesteps=timesteps, folds=folds, results=results,
    )


def run_finalist(
    *,
    root: Path,
    data_root: Path,
    confirm_stage: str,
    confirm_candidate: str,
    hypothesis: str,
    folds_subset: list[str] | None = None,
    seeds_subset: list[int] | None = None,
    timesteps_override: int | None = None,
    registry_path: Path | None = None,
    max_finalists: int = 3,
) -> dict[str, Any]:
    """Retrain one screening candidate from scratch at the finalist budget.

    A finalist confirms the *full effective* configuration of a screening
    candidate (its inherited action set and observation, plus its own change),
    retrained from scratch with the frozen finalist budget (seeds 0-4, 200k
    steps per fold-seed; optimizer and replay state are never reused). It writes
    a ``finalist_confirmation`` record with a distinct id so the screening record
    is untouched. At most ``max_finalists`` finalists may advance, matching the
    frozen contract. Nothing here opens the known period.
    """

    if not hypothesis.strip():
        raise ValueError("a non-empty hypothesis is required before running a finalist")

    fold_contract = load_rolling_folds(root / "configs" / "v2_rolling_folds.yaml")
    search_path = root / "configs" / "dqn_v2_search.yaml"
    search = load_v2_search(search_path)
    if registry_path is None:
        registry_path = root / "experiments" / "v2_registry.jsonl"
    records = load_records(registry_path) if Path(registry_path).exists() else []
    if any(value is not None for value in (folds_subset, seeds_subset, timesteps_override)):
        raise ValueError("formal finalist confirmation forbids fold/seed/budget overrides")
    if confirm_stage != "dqn_hyperparameters":
        raise ValueError("finalists must come from the terminal dqn_hyperparameters stage")

    hp_selection = select_stage_winner(
        records,
        "dqn_hyperparameters",
        **selection_requirements(search, "dqn_hyperparameters"),
    )
    shortlist = [
        row.experiment_id for row in hp_selection.ranking[: int(max_finalists)]
    ]
    source_id = f"v2_{confirm_stage}_{confirm_candidate}"
    if source_id not in shortlist:
        raise ValueError(
            f"{source_id} is not in the automatically frozen top-{max_finalists} shortlist: "
            f"{shortlist}"
        )
    spec = resolve_candidate(search, confirm_stage, confirm_candidate, records=records)

    finalist_id = f"v2_finalist_confirmation_{confirm_stage}_{confirm_candidate}"
    existing_finalists = {
        r["experiment_id"] for r in records if r.get("stage") == "finalist_confirmation"
    }
    if finalist_id in existing_finalists:
        raise ValueError(f"finalist already recorded: {finalist_id}")
    if len(existing_finalists) >= int(max_finalists):
        raise RuntimeError(
            f"finalist cap reached (max {max_finalists}); already advanced: "
            f"{sorted(existing_finalists)}"
        )

    market_path = data_root / "data" / "processed" / "market_base_with_splits.csv"
    event_path = data_root / "data" / "interim" / "distribution_events.csv"
    market = pd.read_csv(market_path)
    events = pd.read_csv(event_path)
    market, events = _development_only(
        market,
        events,
        search["immutable_environment"]["forbidden_selection_period_start"],
    )

    audit = audit_folds(fold_contract, market, events, ticker="SPY")
    if audit.status != "PASS":
        raise RuntimeError(f"fold coverage audit must PASS before training: {audit.to_dict()}")

    seeds, timesteps = budget_and_seeds(search, "finalist")
    if seeds_subset is not None:
        seeds = [s for s in seeds if s in set(seeds_subset)]
    if timesteps_override is not None:
        timesteps = int(timesteps_override)
    folds = [f for f in fold_contract.folds if folds_subset is None or f.fold_id in set(folds_subset)]

    results = _train_spec_over_folds(
        root=root, spec=spec, market=market, events=events,
        fold_contract=fold_contract, folds=folds, seeds=seeds, timesteps=timesteps,
        artifact_id=finalist_id,
    )

    resolved_features = (
        list(spec.market_features) if spec.market_features is not None else list(MARKET_FEATURES)
    )
    changed_fields = {
        "confirms": spec.experiment_id,
        "retrained_from_scratch": True,
        "action_weights": list(spec.action_weights),
        "market_features": resolved_features,
        "include_cash_rate": spec.include_cash_rate,
        "hyperparameters": spec.hyperparameters,
    }
    return _build_record(
        experiment_id=finalist_id,
        stage="finalist_confirmation",
        parent_experiment_id=spec.experiment_id,
        hypothesis=hypothesis,
        changed_category="none_final_confirmation",
        changed_fields=changed_fields,
        root=root, search_path=search_path, market_path=market_path, event_path=event_path,
        seeds=seeds, timesteps=timesteps, folds=folds, results=results,
    )
