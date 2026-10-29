"""Chronological split contract.

The contract is deliberately independent of model code. Training components
may request rows from a named partition, but they must not redefine dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


@dataclass(frozen=True)
class DateRange:
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"invalid date range: {self.start} > {self.end}")

    def contains(self, dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
        values = pd.to_datetime(dates)
        return (values >= self.start) & (values <= self.end)


@dataclass(frozen=True)
class SplitContract:
    train: DateRange
    validation: DateRange
    test: DateRange
    eligible_start: pd.Timestamp
    eligible_end: pd.Timestamp
    warmup_rows: int | None = None
    future_holdout: DateRange | None = None

    def validate(self) -> None:
        ranges = [self.train, self.validation, self.test]
        if self.future_holdout is not None:
            ranges.append(self.future_holdout)

        for left, right in zip(ranges, ranges[1:]):
            if left.end >= right.start:
                raise ValueError(
                    f"split ranges overlap or are out of order: {left} then {right}"
                )

        if self.train.start < self.eligible_start:
            raise ValueError("train starts before eligible_start")
        if ranges[-1].end > self.eligible_end and self.future_holdout is None:
            raise ValueError("test ends after eligible_end")
        if self.warmup_rows is not None and self.warmup_rows < 0:
            raise ValueError("warmup_rows must be non-negative or null")

    def range_for(self, partition: str) -> DateRange:
        if partition not in {"train", "validation", "test", "future_holdout"}:
            raise KeyError(f"unknown partition: {partition}")
        value = getattr(self, partition)
        if value is None:
            raise ValueError(f"partition is not configured: {partition}")
        return value

    def select(
        self,
        frame: pd.DataFrame,
        partition: str,
        *,
        date_column: str = "date",
    ) -> pd.DataFrame:
        if date_column not in frame.columns:
            raise KeyError(f"missing date column: {date_column}")
        selected_range = self.range_for(partition)
        mask = selected_range.contains(frame[date_column])
        return frame.loc[mask].copy()


def _parse_range(value: Mapping[str, Any] | None) -> DateRange | None:
    if value is None:
        return None
    return DateRange(pd.Timestamp(value["start"]), pd.Timestamp(value["end"]))


def load_split_contract(path: str | Path) -> SplitContract:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    contract = SplitContract(
        train=_parse_range(raw["train"]),  # type: ignore[arg-type]
        validation=_parse_range(raw["validation"]),  # type: ignore[arg-type]
        test=_parse_range(raw["test"]),  # type: ignore[arg-type]
        eligible_start=pd.Timestamp(raw["eligible_start"]),
        eligible_end=pd.Timestamp(raw["eligible_end"]),
        warmup_rows=raw.get("warmup_rows"),
        future_holdout=_parse_range(raw.get("future_holdout")),
    )
    contract.validate()
    return contract

