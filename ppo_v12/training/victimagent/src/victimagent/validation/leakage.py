"""Reusable leakage assertions for the eventual production feature builder."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd
from pandas.testing import assert_frame_equal

FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]


def _prefix(frame: pd.DataFrame, cutoff: pd.Timestamp, date_column: str) -> pd.DataFrame:
    if date_column not in frame.columns:
        raise KeyError(f"missing date column: {date_column}")
    dates = pd.to_datetime(frame[date_column])
    return frame.loc[dates <= cutoff].reset_index(drop=True)


def assert_future_invariance(
    builder: FeatureBuilder,
    original: pd.DataFrame,
    modified_future: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    *,
    date_column: str = "date",
) -> None:
    """Changing rows after cutoff must not change features at or before cutoff."""

    cutoff_ts = pd.Timestamp(cutoff)
    before = _prefix(builder(original.copy()), cutoff_ts, date_column)
    after = _prefix(builder(modified_future.copy()), cutoff_ts, date_column)
    assert_frame_equal(before, after, check_exact=True)


def assert_prefix_equivalence(
    builder: FeatureBuilder,
    frame: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    *,
    date_column: str = "date",
) -> None:
    """Full-history and prefix-only calculations must match through cutoff."""

    cutoff_ts = pd.Timestamp(cutoff)
    full_result = _prefix(builder(frame.copy()), cutoff_ts, date_column)
    input_prefix = _prefix(frame, cutoff_ts, date_column)
    prefix_result = builder(input_prefix).reset_index(drop=True)
    assert_frame_equal(full_result, prefix_result, check_exact=True)


def assert_window_boundary(
    *,
    episode_start_position: int,
    earliest_read_position: int,
    warmup_rows: int,
) -> None:
    """Reject feature reads earlier than the declared pre-episode warm-up."""

    if warmup_rows < 0:
        raise ValueError("warmup_rows must be non-negative")
    allowed_earliest = episode_start_position - warmup_rows
    if earliest_read_position < allowed_earliest:
        raise AssertionError(
            "feature builder read before the allowed warm-up boundary: "
            f"{earliest_read_position} < {allowed_earliest}"
        )


def assert_forbidden_columns_absent(
    requested_columns: Iterable[str],
    *,
    forbidden_columns: Iterable[str] = (
        "future_return",
        "next_return",
        "target",
        "label",
        "next_action",
    ),
) -> None:
    """Prevent the feature builder from requesting labels or future outcomes."""

    requested = set(requested_columns)
    forbidden = requested.intersection(forbidden_columns)
    if forbidden:
        raise AssertionError(f"forbidden feature inputs requested: {sorted(forbidden)}")

