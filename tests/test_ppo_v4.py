from pathlib import Path

import pytest

from victim_ppo.config import V4_COMMIT, load_config
from victim_ppo.runner import linear_schedule
from scripts.run_ppo_clean_benchmark import aggregate_reports


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_shared_contract_and_formal_budget():
    config = load_config(ROOT / "configs/ppo_v4.yaml")
    assert config.raw["status"] == "frozen_before_ppo_training"
    assert config.raw["shared_contract"]["victimagent_commit"] == V4_COMMIT
    assert config.raw["shared_contract"]["observation_dimensions"] == 10
    assert config.formal["seeds"] == [0, 1, 2, 3, 4]
    assert config.formal["total_timesteps_per_seed"] == 200_000


def test_linear_learning_rate_descends_to_declared_floor():
    schedule = linear_schedule(3e-4, 3e-5)
    assert schedule(1.0) == pytest.approx(3e-4)
    assert schedule(0.5) == pytest.approx(1.65e-4)
    assert schedule(0.0) == pytest.approx(3e-5)
    assert schedule(1.0) > schedule(0.5) > schedule(0.0)


def test_aggregate_rejects_missing_seed_reports(tmp_path):
    (tmp_path / "reports").mkdir()
    with pytest.raises(FileNotFoundError, match="SEED0"):
        aggregate_reports(tmp_path)

