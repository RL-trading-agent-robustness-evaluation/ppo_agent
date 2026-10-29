"""Research-validity checks."""

from .leakage import (
    assert_forbidden_columns_absent,
    assert_future_invariance,
    assert_prefix_equivalence,
    assert_window_boundary,
)

__all__ = [
    "assert_forbidden_columns_absent",
    "assert_future_invariance",
    "assert_prefix_equivalence",
    "assert_window_boundary",
]

