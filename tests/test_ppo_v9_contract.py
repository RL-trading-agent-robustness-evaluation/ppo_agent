from pathlib import Path

import pytest

from victim_ppo.v9.config import (ARMS, EXPECTED_ACTUAL_TIMESTEPS, EXPECTED_PIN,
                                  RAW_FILES, load_config, missing_raw_files)
from victim_ppo.v9.runner import linear_schedule, preflight, write_new

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ppo_v9_reward_ablation.yaml"


def test_frozen_v9_contract_and_rollout_overshoot():
    config = load_config(CONFIG)
    assert config.raw["shared_contract"]["victimagent_commit"] == EXPECTED_PIN
    assert tuple(config.raw["arms"]) == ARMS
    assert config.raw["formal"]["expected_sb3_timesteps_per_run"] == EXPECTED_ACTUAL_TIMESTEPS
    assert config.raw["formal"]["record_revision"] == "v9spec1"
    assert config.raw["formal"]["final_arm"] == "v9_gate_relative_reward"
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


def test_records_are_append_only(tmp_path):
    path = tmp_path / "record.json"
    write_new(path, {"status": "reserved"})
    with pytest.raises(FileExistsError):
        write_new(path, {"status": "overwritten"})


def test_final_and_confirmation_contract_is_frozen():
    config = load_config(CONFIG)
    assert tuple(config.raw["final"]["seeds"]) == (0, 1, 2, 3, 4)
    assert config.raw["final"]["requested_timesteps_per_seed"] == 500_000
    assert config.raw["final"]["known_period"]["one_shot"] is True
