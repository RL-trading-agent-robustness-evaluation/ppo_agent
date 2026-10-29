"""V2 rolling-origin development folds and the pre-training coverage audit.

The folds are frozen in ``configs/v2_rolling_folds.yaml`` before any V2 result
exists. This module loads them, turns each into an ordinary ``SplitContract``
(so the existing train-only scaler and bundle code apply unchanged, one fold at
a time), and runs the ``required_audit_before_training`` battery. No fold may be
used for training until the audit passes, and no fold ever reaches into the
2022-2025 known-period window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from victimagent.data.splits import DateRange, SplitContract

# A 504-decision episode consumes 504 decisions plus two future opens for the
# final reward, so the training window needs at least this many sessions.
EPISODE_DECISIONS = 504
EPISODE_SESSION_FLOOR = EPISODE_DECISIONS + 2


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train: DateRange
    validation: DateRange


@dataclass(frozen=True)
class RollingFoldContract:
    version: int
    warmup_rows: int
    folds: tuple[Fold, ...]
    development: DateRange
    known_period: DateRange
    forbidden_selection_start: pd.Timestamp

    def split_contract_for(self, fold: Fold) -> SplitContract:
        """A one-fold ``SplitContract``.

        ``test`` is pinned to the known-period window purely as a placeholder so
        the contract validates; V2 selection never selects that partition. The
        scaler fits on ``train`` (this fold's train) exactly as in V1.
        """

        contract = SplitContract(
            train=fold.train,
            validation=fold.validation,
            test=self.known_period,
            eligible_start=self.development.start
            if self.warmup_rows == 0
            else pd.Timestamp(f"{self.development.start.year - 1}-01-01"),
            eligible_end=self.known_period.end,
            warmup_rows=self.warmup_rows,
        )
        contract.validate()
        return contract


def _range(node: dict[str, Any]) -> DateRange:
    return DateRange(pd.Timestamp(node["start"]), pd.Timestamp(node["end"]))


def load_rolling_folds(path: str | Path) -> RollingFoldContract:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    roles = raw["data_roles"]
    folds = tuple(
        Fold(
            fold_id=str(item["id"]),
            train=_range(item["train"]),
            validation=_range(item["validation"]),
        )
        for item in raw["folds"]
    )
    forbidden = pd.Timestamp(roles["known_period_confirmation"]["start"])
    return RollingFoldContract(
        version=int(raw["version"]),
        warmup_rows=int(raw["warmup_rows"]),
        folds=folds,
        development=_range(roles["development"]),
        known_period=_range(roles["known_period_confirmation"]),
        forbidden_selection_start=forbidden,
    )


@dataclass(frozen=True)
class FoldAuditResult:
    status: str
    per_fold: dict[str, dict[str, bool]]
    global_checks: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "global_checks": self.global_checks,
            "per_fold": self.per_fold,
        }


def _sessions(market: pd.DataFrame, ticker: str) -> pd.DatetimeIndex:
    asset = market.loc[market["ticker"].astype(str).str.upper() == ticker.upper()]
    dates = pd.to_datetime(asset["date"]).dt.normalize().drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


def _spy_ex_dividends(events: pd.DataFrame, ticker: str) -> pd.DatetimeIndex:
    if events.empty or "ticker" not in events.columns:
        return pd.DatetimeIndex([])
    rows = events.loc[
        (events["ticker"].astype(str).str.upper() == ticker.upper())
        & (events["disordinaryflg"].astype(str).str.upper() == "Y")
    ]
    if rows.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(rows["disexdt"]).dt.normalize())


def audit_folds(
    contract: RollingFoldContract,
    market: pd.DataFrame,
    events: pd.DataFrame,
    *,
    ticker: str = "SPY",
) -> FoldAuditResult:
    """Run every ``required_audit_before_training`` check on real sessions."""

    sessions = _sessions(market, ticker)
    ex_dates = _spy_ex_dividends(events, ticker)

    train_starts = {fold.train.start for fold in contract.folds}
    validation_ranges = sorted(
        (fold.validation.start, fold.validation.end) for fold in contract.folds
    )
    validations_disjoint = all(
        earlier_end < later_start
        for (_, earlier_end), (later_start, _) in zip(validation_ranges, validation_ranges[1:])
    )
    earliest_train_start = min(fold.train.start for fold in contract.folds)
    latest_validation_end = max(fold.validation.end for fold in contract.folds)

    # Cash is an observation/accounting input on formal train and validation
    # sessions.  Pre-development warm-up rows exist to build lagged market
    # features and are not themselves decision rows, so a missing FRED value
    # there must not block Gate C0.  Conversely, every SPY row in the frozen
    # 2010-2021 selection span must have the already-lagged cash rate.
    asset_mask = market["ticker"].astype(str).str.upper() == ticker.upper()
    market_dates = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    selection_mask = (
        asset_mask
        & market_dates.between(earliest_train_start, latest_validation_end)
    )
    cash_rate = pd.to_numeric(
        market.loc[selection_mask, "dgs3mo_pct"], errors="coerce"
    )
    finite_cash = bool(cash_rate.notna().all()) and len(cash_rate) > 0

    global_checks = {
        "dates_parse": True,
        "expanding_train_start_is_constant": len(train_starts) == 1,
        "validation_windows_do_not_overlap": validations_disjoint,
        "all_selection_dates_end_before_2022": latest_validation_end
        < contract.forbidden_selection_start,
        "lagged_cash_rate_is_available": bool(finite_cash),
    }

    per_fold: dict[str, dict[str, bool]] = {}
    for fold in contract.folds:
        in_train = (sessions >= fold.train.start) & (sessions <= fold.train.end)
        in_val = (sessions >= fold.validation.start) & (sessions <= fold.validation.end)
        train_count = int(in_train.sum())
        before_train = int((sessions < fold.train.start).sum())
        ex_in_span = int(
            (
                (ex_dates >= fold.train.start) & (ex_dates <= fold.validation.end)
            ).sum()
        )
        per_fold[fold.fold_id] = {
            "train_precedes_validation": fold.train.end < fold.validation.start,
            "no_train_validation_overlap": not bool((in_train & in_val).any()),
            "each_fold_has_at_least_one_full_504_decision_training_episode": train_count
            >= EPISODE_SESSION_FLOOR,
            "warmup_coverage_is_available": before_train >= contract.warmup_rows,
            "corporate_action_coverage_is_available": ex_in_span >= 1,
            "validation_has_sessions": bool(in_val.any()),
        }

    all_pass = all(global_checks.values()) and all(
        all(checks.values()) for checks in per_fold.values()
    )
    return FoldAuditResult(
        status="PASS" if all_pass else "FAIL",
        per_fold=per_fold,
        global_checks=global_checks,
    )
