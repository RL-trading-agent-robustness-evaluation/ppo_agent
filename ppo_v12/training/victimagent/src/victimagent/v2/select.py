"""Gate E0 selection with strict completeness and provenance guards.

Selection is pure registry arithmetic.  A stage is selectable only after every
candidate declared by the frozen contract has a complete, internally consistent
fold x seed result matrix at the frozen budget.  Smoke/partial records therefore
cannot become research winners.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateSummary:
    experiment_id: str
    stage: str
    median_sharpe_excess_lagged_cash: float | None
    median_annualized_excess_return: float | None
    median_max_drawdown_magnitude: float | None
    median_turnover: float | None
    eligible_fold_seeds: int

    def sort_key(self) -> tuple[float, float, float, float, str]:
        neg_inf = -math.inf
        pos_inf = math.inf
        sharpe = self.median_sharpe_excess_lagged_cash
        excess = self.median_annualized_excess_return
        drawdown = self.median_max_drawdown_magnitude
        turnover = self.median_turnover
        return (
            -(sharpe if sharpe is not None else neg_inf),
            -(excess if excess is not None else neg_inf),
            drawdown if drawdown is not None else pos_inf,
            turnover if turnover is not None else pos_inf,
            self.experiment_id,
        )


@dataclass(frozen=True)
class StageSelection:
    stage: str
    winner_experiment_id: str
    ranking: tuple[CandidateSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "winner_experiment_id": self.winner_experiment_id,
            "primary_metric": "median_sharpe_excess_lagged_cash",
            "tie_breakers": [
                "higher_median_annualized_excess_return",
                "smaller_median_max_drawdown_magnitude",
                "lower_median_turnover",
                "lexical_experiment_id",
            ],
            "ranking": [
                {
                    "experiment_id": s.experiment_id,
                    "median_sharpe_excess_lagged_cash": s.median_sharpe_excess_lagged_cash,
                    "median_annualized_excess_return": s.median_annualized_excess_return,
                    "median_max_drawdown_magnitude": s.median_max_drawdown_magnitude,
                    "median_turnover": s.median_turnover,
                    "eligible_fold_seeds": s.eligible_fold_seeds,
                }
                for s in self.ranking
            ],
        }


def _median(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.median(clean)) if clean else None


def summarize_record(record: dict[str, Any]) -> CandidateSummary:
    metrics = record.get("validation_metrics") or {}
    rows = [row for row in metrics.get("per_fold_seed", []) if row.get("eligible")]
    return CandidateSummary(
        experiment_id=record["experiment_id"],
        stage=record["stage"],
        median_sharpe_excess_lagged_cash=_median(
            [row.get("sharpe_excess_cash") for row in rows]
        ),
        median_annualized_excess_return=_median(
            [row.get("annualized_mean_excess_return") for row in rows]
        ),
        median_max_drawdown_magnitude=_median(
            [abs(row["max_drawdown"]) for row in rows if row.get("max_drawdown") is not None]
        ),
        median_turnover=_median([row.get("total_turnover") for row in rows]),
        eligible_fold_seeds=len(rows),
    )


def _selectable(record: dict[str, Any], stage: str) -> bool:
    return (
        record.get("stage") == stage
        and record.get("status") == "pass"
        and record.get("known_period_accessed") is False
    )


def _provenance_key(record: dict[str, Any]) -> tuple[Any, ...]:
    data = record.get("data_sha256") or {}
    return (
        record.get("git_commit"),
        record.get("config_path"),
        record.get("config_sha256"),
        tuple(sorted(data.items())),
        record.get("split_sha256"),
    )


def validate_stage_completeness(
    records: list[dict[str, Any]],
    stage: str,
    *,
    expected_experiment_ids: list[str] | tuple[str, ...],
    expected_folds: list[str] | tuple[str, ...],
    expected_seeds: list[int] | tuple[int, ...],
    expected_timesteps: int,
) -> list[dict[str, Any]]:
    """Return the exact formal stage records or raise before ranking.

    This is deliberately stricter than the JSON schema: schemas describe shape,
    while this guard enforces the frozen research design.
    """

    expected_ids = set(expected_experiment_ids)
    stage_records = [r for r in records if r.get("stage") == stage]
    ids = [r.get("experiment_id") for r in stage_records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate experiment ids in stage '{stage}'")
    actual_ids = set(ids)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            f"incomplete stage '{stage}': missing={missing}, unexpected={extra}"
        )

    expected_folds_tuple = tuple(expected_folds)
    expected_seeds_tuple = tuple(int(s) for s in expected_seeds)
    expected_pairs = {
        (fold, seed) for fold in expected_folds_tuple for seed in expected_seeds_tuple
    }
    provenance: set[tuple[Any, ...]] = set()

    for record in stage_records:
        experiment_id = record["experiment_id"]
        if record.get("known_period_accessed") is not False:
            raise ValueError(f"{experiment_id}: known period was accessed")
        if tuple(record.get("folds") or ()) != expected_folds_tuple:
            raise ValueError(f"{experiment_id}: incomplete or reordered fold set")
        if tuple(record.get("seeds") or ()) != expected_seeds_tuple:
            raise ValueError(f"{experiment_id}: incomplete or reordered seed set")
        budget = (record.get("training_budget") or {}).get("timesteps_per_fold_seed")
        if int(budget or -1) != int(expected_timesteps):
            raise ValueError(f"{experiment_id}: wrong training budget")

        rows = (record.get("validation_metrics") or {}).get("per_fold_seed") or []
        pairs = [(row.get("fold"), row.get("seed")) for row in rows]
        if len(pairs) != len(set(pairs)) or set(pairs) != expected_pairs:
            raise ValueError(f"{experiment_id}: incomplete fold x seed result matrix")
        if record.get("status") == "pass" and not all(
            row.get("eligible") is True for row in rows
        ):
            raise ValueError(f"{experiment_id}: PASS conflicts with ineligible rows")
        provenance.add(_provenance_key(record))

    if len(provenance) != 1:
        raise ValueError(
            f"stage '{stage}' mixes git/config/data/split provenance; restart the stage"
        )
    return stage_records


def select_stage_winner(
    records: list[dict[str, Any]],
    stage: str,
    *,
    expected_experiment_ids: list[str] | tuple[str, ...] | None = None,
    expected_folds: list[str] | tuple[str, ...] | None = None,
    expected_seeds: list[int] | tuple[int, ...] | None = None,
    expected_timesteps: int | None = None,
) -> StageSelection:
    """Apply the frozen rule after optional strict formal-stage validation."""

    strict_args = (
        expected_experiment_ids,
        expected_folds,
        expected_seeds,
        expected_timesteps,
    )
    if any(value is not None for value in strict_args):
        if any(value is None for value in strict_args):
            raise ValueError("strict selection requires ids, folds, seeds, and timesteps")
        population = validate_stage_completeness(
            records,
            stage,
            expected_experiment_ids=expected_experiment_ids,
            expected_folds=expected_folds,
            expected_seeds=expected_seeds,
            expected_timesteps=int(expected_timesteps),
        )
    else:
        population = records

    candidates = [summarize_record(r) for r in population if _selectable(r, stage)]
    if not candidates:
        raise ValueError(
            f"no selectable (status=='pass') records for stage '{stage}'; "
            "complete the frozen stage and retain all failures"
        )
    ranking = tuple(sorted(candidates, key=CandidateSummary.sort_key))
    return StageSelection(
        stage=stage,
        winner_experiment_id=ranking[0].experiment_id,
        ranking=ranking,
    )
