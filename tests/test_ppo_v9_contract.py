from pathlib import Path

import pytest

from victim_ppo.v9.config import (ARMS, EXPECTED_ACTUAL_TIMESTEPS, EXPECTED_PIN,
                                  RAW_FILES, load_config, missing_raw_files)
from victim_ppo.v9.runner import linear_schedule, preflight

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ppo_v9_reward_ablation.yaml"


def test_frozen_v9_contract_and_rollout_overshoot():
    config = load_config(CONFIG)
    assert config.raw["shared_contract"]["victimagent_commit"] == EXPECTED_PIN
    assert tuple(config.raw["arms"]) == ARMS
    assert config.raw["formal"]["expected_sb3_timesteps_per_run"] == EXPECTED_ACTUAL_TIMESTEPS
    assert config.raw["formal"]["record_revision"] == "v9spec1"
    assert config.upstream_arm(ARMS[0]) == "v9_trend_gate"
    assert config.upstream_arm(ARMS[1]) == "v9_gate_relative_reward"


def test_learning_rate_descends_and_clamps():
    schedule = linear_schedule(3e-4, 3e-5)
    assert schedule(1.0) == pytest.approx(3e-4)
    assert schedule(0.5) == pytest.approx(1.65e-4)
    assert schedule(0.0) == pytest.approx(3e-5)
    assert schedule(-1.0) == pytest.approx(3e-5)


def test_empty_raw_directory_inventory(tmp_path):
    config = load_config(CONFIG)
    assert missing_raw_files(tmp_path / "victimagent") == list(RAW_FILES)
    result = preflight(ROOT, config, require_data=True)
    assert result["status"] == "PASS"
    assert result["known_period_accessed"] is False
    assert result["missing_raw_files"] == []
