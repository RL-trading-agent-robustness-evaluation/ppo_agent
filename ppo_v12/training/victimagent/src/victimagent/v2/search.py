"""Resolve one V2 candidate from the frozen search contract.

``configs/dqn_v2_search.yaml`` is frozen before results. This module turns a
``(stage, candidate_id)`` pair into a concrete, auditable spec: the action
representation, the observation layout, the DQN hyperparameters, the parent
experiment, and the single changed category. It refuses to invent candidates the
contract did not declare.

The study is staged one factor at a time, so each stage inherits the *winning*
configuration of the previous stage:

    action_space          parent = dqn_v1 baseline (ternary, full obs, baseline HP)
    observation_ablation  parent = winner of action_space  (inherits its action set)
    dqn_hyperparameters   parent = winner of observation_ablation (inherits action + obs)

Inheritance is resolved from the append-only registry via the frozen Gate E0
rule (:mod:`victimagent.v2.select`). Resolving an ``observation_ablation`` or
``dqn_hyperparameters`` candidate therefore requires the prior stage's completed
records; ``action_space`` needs none. Nothing here reads the known period.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from victimagent.features import canonical_market_subset
from victimagent.v2.select import select_stage_winner


@dataclass(frozen=True)
class CandidateSpec:
    experiment_id: str
    stage: str
    candidate_id: str
    parent_experiment_id: str | None
    changed_category: str
    changed_fields: dict[str, Any]
    action_weights: tuple[float, ...]
    hyperparameters: dict[str, Any]
    # None => the frozen full V1 market feature set. A tuple names the ablated
    # market feature subset (observation_ablation stage onward).
    market_features: tuple[str, ...] | None
    include_cash_rate: bool = True


@dataclass(frozen=True)
class ParentContext:
    """The effective configuration a stage inherits from its parent."""

    experiment_id: str
    action_weights: tuple[float, ...]
    market_features: tuple[str, ...] | None
    include_cash_rate: bool
    hyperparameters: dict[str, Any]


_STAGE_CHANGED_CATEGORY = {
    "action_space": "action_space",
    "observation_ablation": "observation",
    "dqn_hyperparameters": "hyperparameter",
    "finalist_confirmation": "none_final_confirmation",
}

_V1_ACTION_WEIGHTS = (0.0, 0.5, 1.0)


def load_v2_search(path: str | Path) -> dict:
    """Load the frozen search plus the frozen fold ids used by strict selection."""

    search_path = Path(path)
    raw = yaml.safe_load(search_path.read_text(encoding="utf-8"))
    fold_path = search_path.with_name("v2_rolling_folds.yaml")
    folds = yaml.safe_load(fold_path.read_text(encoding="utf-8"))
    raw["_expected_fold_ids"] = [item["id"] for item in folds["folds"]]
    return raw


def declared_candidate_ids(raw: dict, stage_id: str) -> list[str]:
    """Return every frozen experiment id that must exist before selection."""

    stage = _stage(raw, stage_id)
    if stage_id == "dqn_hyperparameters":
        candidate_ids = list(enumerate_hp_candidates(stage))
    else:
        candidate_ids = [item["id"] for item in stage["candidates"]]
    return [f"v2_{stage_id}_{candidate_id}" for candidate_id in candidate_ids]


def selection_requirements(
    raw: dict,
    stage_id: str,
    *,
    expected_experiment_ids: list[str] | None = None,
    mode: str = "screening",
) -> dict[str, Any]:
    """Build the exact Gate E0 completeness contract for one stage."""

    seeds, timesteps = budget_and_seeds(raw, mode)
    fold_ids = raw.get("_expected_fold_ids")
    if not fold_ids:
        raise ValueError("search config was not loaded with its frozen fold contract")
    return {
        "expected_experiment_ids": (
            expected_experiment_ids
            if expected_experiment_ids is not None
            else declared_candidate_ids(raw, stage_id)
        ),
        "expected_folds": list(fold_ids),
        "expected_seeds": seeds,
        "expected_timesteps": timesteps,
    }


def baseline_hyperparameters(raw: dict) -> dict[str, Any]:
    """The frozen DQN baseline: search baseline merged with its fixed block."""

    stage = next(s for s in raw["stages"] if s["id"] == "dqn_hyperparameters")
    merged = dict(stage["fixed"])
    merged.update(stage["baseline"])
    merged.setdefault("exploration_initial_eps", 1.0)
    merged.setdefault("activation_fn", "ReLU")
    return merged


def budget_and_seeds(raw: dict, mode: str) -> tuple[list[int], int]:
    if mode == "screening":
        block = raw["screening"]
        return list(block["seeds"]), int(block["total_timesteps_per_fold_seed"])
    if mode == "finalist":
        block = raw["finalist_confirmation"]
        return list(block["seeds"]), int(block["total_timesteps_per_fold_seed"])
    raise ValueError("mode must be 'screening' or 'finalist'")


def _stage(raw: dict, stage_id: str) -> dict:
    for stage in raw["stages"]:
        if stage["id"] == stage_id:
            return stage
    raise KeyError(f"unknown stage: {stage_id}")


def _slug(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "_".join(str(int(v)) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".10f").rstrip("0").rstrip(".")
    return str(value)


def enumerate_hp_candidates(stage: dict) -> dict[str, tuple[str, Any]]:
    """Return the unchanged parent control plus one-axis-at-a-time candidates.

    The explicit control lets Gate E0 retain the inherited parent when every
    modified candidate is worse. Axis values equal to the baseline are skipped
    because the control already represents that effective configuration.
    """

    baseline = stage["baseline"]
    axes = stage["axes"]
    candidates: dict[str, tuple[str, Any]] = {}
    control = stage.get("control_candidate")
    if control:
        control_id = str(control["id"])
        if not bool(control.get("inherit_parent_hyperparameters_unchanged")):
            raise ValueError("HP control candidate must inherit parent hyperparameters")
        candidates[control_id] = ("__parent_baseline__", None)
    for axis, values in axes.items():
        for value in values:
            if value == baseline.get(axis):
                continue
            candidate_id = f"{axis}__{_slug(value)}"
            candidates[candidate_id] = (axis, value)
    return candidates


def _dqn_v1_context(raw: dict) -> ParentContext:
    return ParentContext(
        experiment_id="dqn_v1",
        action_weights=_V1_ACTION_WEIGHTS,
        market_features=None,
        include_cash_rate=True,
        hyperparameters=baseline_hyperparameters(raw),
    )


def _winner_context(
    raw: dict, records: list[dict[str, Any]], stage_id: str
) -> ParentContext:
    selection = select_stage_winner(
        records,
        stage_id,
        **selection_requirements(raw, stage_id),
    )
    winner_id = selection.winner_experiment_id
    prefix = f"v2_{stage_id}_"
    if not winner_id.startswith(prefix):
        raise ValueError(f"winner id {winner_id!r} is not a '{stage_id}' experiment")
    winner_candidate = winner_id[len(prefix):]
    spec = resolve_candidate(raw, stage_id, winner_candidate, records=records)
    return ParentContext(
        experiment_id=winner_id,
        action_weights=spec.action_weights,
        market_features=spec.market_features,
        include_cash_rate=spec.include_cash_rate,
        hyperparameters=spec.hyperparameters,
    )


def resolve_candidate(
    raw: dict,
    stage_id: str,
    candidate_id: str,
    *,
    records: list[dict[str, Any]] | None = None,
) -> CandidateSpec:
    stage = _stage(raw, stage_id)
    changed_category = _STAGE_CHANGED_CATEGORY[stage_id]

    if stage_id == "action_space":
        parent = _dqn_v1_context(raw)
        candidates = {item["id"]: item for item in stage["candidates"]}
        if candidate_id not in candidates:
            raise KeyError(
                f"candidate '{candidate_id}' not declared in stage '{stage_id}': "
                f"{sorted(candidates)}"
            )
        action_weights = tuple(float(w) for w in candidates[candidate_id]["action_weights"])
        return CandidateSpec(
            experiment_id=f"v2_{stage_id}_{candidate_id}",
            stage=stage_id,
            candidate_id=candidate_id,
            parent_experiment_id=parent.experiment_id,
            changed_category=changed_category,
            changed_fields={"action_weights": list(action_weights)},
            action_weights=action_weights,
            hyperparameters=parent.hyperparameters,
            market_features=parent.market_features,
            include_cash_rate=parent.include_cash_rate,
        )

    if stage_id == "observation_ablation":
        if records is None:
            raise ValueError(
                "observation_ablation resolution needs the prior stage's registry "
                "records (to inherit the action_space winner); pass records=..."
            )
        parent = _winner_context(raw, records, "action_space")
        candidates = {item["id"]: item for item in stage["candidates"]}
        if candidate_id not in candidates:
            raise KeyError(
                f"candidate '{candidate_id}' not declared in stage '{stage_id}': "
                f"{sorted(candidates)}"
            )
        candidate = candidates[candidate_id]
        if not bool(candidate.get("include_account_state", True)):
            raise NotImplementedError(
                "dropping the account-state fractions is out of the frozen V2 scope"
            )
        market_features = canonical_market_subset(candidate["market_features"])
        include_cash_rate = bool(candidate.get("include_cash_rate", True))
        return CandidateSpec(
            experiment_id=f"v2_{stage_id}_{candidate_id}",
            stage=stage_id,
            candidate_id=candidate_id,
            parent_experiment_id=parent.experiment_id,
            changed_category=changed_category,
            changed_fields={
                "market_features": list(market_features),
                "include_cash_rate": include_cash_rate,
            },
            action_weights=parent.action_weights,
            hyperparameters=parent.hyperparameters,
            market_features=market_features,
            include_cash_rate=include_cash_rate,
        )

    if stage_id == "dqn_hyperparameters":
        if records is None:
            raise ValueError(
                "dqn_hyperparameters resolution needs the prior stage's registry "
                "records (to inherit the observation_ablation winner); pass records=..."
            )
        parent = _winner_context(raw, records, "observation_ablation")
        candidates = enumerate_hp_candidates(stage)
        if candidate_id not in candidates:
            raise KeyError(
                f"candidate '{candidate_id}' not declared in stage '{stage_id}': "
                f"{sorted(candidates)}"
            )
        axis, value = candidates[candidate_id]
        hyperparameters = dict(parent.hyperparameters)
        if axis == "__parent_baseline__":
            changed_fields = {
                "control": "inherited_parent_hyperparameters_unchanged"
            }
        else:
            hyperparameters[axis] = (
                list(value) if isinstance(value, (list, tuple)) else value
            )
            changed_fields = {axis: value}
        return CandidateSpec(
            experiment_id=f"v2_{stage_id}_{candidate_id}",
            stage=stage_id,
            candidate_id=candidate_id,
            parent_experiment_id=parent.experiment_id,
            changed_category=changed_category,
            changed_fields=changed_fields,
            action_weights=parent.action_weights,
            hyperparameters=hyperparameters,
            market_features=parent.market_features,
            include_cash_rate=parent.include_cash_rate,
        )

    raise KeyError(f"unknown or non-runnable stage: {stage_id}")
