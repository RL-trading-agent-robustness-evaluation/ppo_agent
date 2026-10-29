import pytest

from scripts.evaluate_ppo_v9_known_diagnostic import distribution


def test_distribution_reports_median_iqr_and_bounds():
    result = distribution([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == {"median": 3.0, "iqr": 2.0, "min": 1.0, "max": 5.0}


def test_distribution_accepts_negative_gaps():
    result = distribution([-0.3, -0.1, 0.0, 0.2, 0.4])
    assert result["median"] == 0.0
    assert result["iqr"] == pytest.approx(0.3)

