from pathlib import Path

from scripts.run_ppo_v10_pilot import load_pilot
from victim_ppo.v10.config import ARMS, SEEDS

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_contract_is_paired_and_small():
    pilot = load_pilot(ROOT / "configs/ppo_v10_pilot_200k.yaml")
    assert pilot["fold"] == "fold_1"
    assert tuple(pilot["arms"]) == ARMS
    assert tuple(pilot["seeds"]) == SEEDS
    assert pilot["requested_timesteps_per_run"] == 200_000
    assert pilot["expected_sb3_timesteps_per_run"] == 200_704
    assert pilot["known_period_accessed"] is False
    assert pilot["selection_effect"] == "none"

