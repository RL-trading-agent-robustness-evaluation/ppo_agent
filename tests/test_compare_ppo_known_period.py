import pytest

from scripts.compare_ppo_known_period_v4_v9_v10 import distribution, verify_boundary


def test_distribution_reports_seed_spread():
    rows = [{"x": value} for value in (1, 2, 3, 4, 5)]
    assert distribution(rows, "x") == {"median": 3.0, "iqr": 2.0, "min": 1.0, "max": 5.0}


def test_verify_boundary_rejects_mismatch():
    with pytest.raises(RuntimeError, match="boundary"):
        verify_boundary({"steps": 999, "start_date": "2022-01-03",
                         "end_date": "2025-12-31"}, "bad")
