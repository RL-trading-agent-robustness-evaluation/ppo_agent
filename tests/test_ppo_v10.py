from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from victim_ppo.v10.config import (ARMS, BETA, STREAK_CAP,
                                   EXPECTED_ACTUAL_TIMESTEPS, load_config)
from victim_ppo.v10.wrappers import PositiveStreakRewardWrapper

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ppo_v10_nav_trend_reward.yaml"


class RewardSequenceEnv(gym.Env):
    action_space = gym.spaces.Discrete(3)
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self, rewards):
        self.rewards = list(rewards)
        self.index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        reward = self.rewards[self.index]
        self.index += 1
        terminated = self.index == len(self.rewards)
        return np.zeros(1, dtype=np.float32), reward, terminated, False, {}


def test_v10_contract_is_frozen_and_keeps_v9_matrix():
    config = load_config(CONFIG)
    assert tuple(config.raw["arms"]) == ARMS
    assert config.raw["streak_reward"]["beta"] == BETA
    assert config.raw["streak_reward"]["streak_cap"] == STREAK_CAP
    assert config.raw["formal"]["expected_sb3_timesteps_per_run"] == EXPECTED_ACTUAL_TIMESTEPS
    assert config.uses_streak_reward(ARMS[0]) is False
    assert config.uses_streak_reward(ARMS[1]) is True
    assert config.raw["known_period"]["enabled_by_this_runner"] is False


def test_positive_streak_formula_breaks_on_zero_and_negative():
    env = PositiveStreakRewardWrapper(
        RewardSequenceEnv([0.01, 0.02, 0.03, 0.0, -0.01, 0.04]),
        beta=0.10, streak_cap=5, enabled=True,
    )
    env.reset()
    expected = [0.01, 0.022, 0.036, 0.0, -0.01, 0.04]
    streaks = [1, 2, 3, 0, 0, 1]
    for reward_expected, streak_expected in zip(expected, streaks):
        _, reward, _, _, info = env.step(0)
        assert reward == pytest.approx(reward_expected)
        assert info["positive_return_streak"] == streak_expected
        assert info["training_reward"] == reward


def test_streak_multiplier_caps_and_reset_clears_state():
    env = PositiveStreakRewardWrapper(RewardSequenceEnv([0.01] * 7), enabled=True)
    env.reset()
    rewards = [env.step(0)[1] for _ in range(7)]
    assert rewards == pytest.approx([0.010, 0.011, 0.012, 0.013, 0.014, 0.014, 0.014])
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert reward == pytest.approx(0.01)
    assert info["positive_return_streak"] == 1


def test_control_exposes_diagnostics_without_bonus():
    env = PositiveStreakRewardWrapper(RewardSequenceEnv([0.01, 0.02]), enabled=False)
    env.reset()
    env.step(0)
    _, reward, _, _, info = env.step(0)
    assert info["positive_return_streak"] == 2
    assert info["positive_streak_bonus"] == 0.0
    assert reward == pytest.approx(0.02)
    assert info["nav_log_reward"] == pytest.approx(0.02)


def test_reward_identity_on_complete_episode():
    nav_rewards = [0.01, 0.02, -0.01, 0.03, 0.04]
    env = PositiveStreakRewardWrapper(RewardSequenceEnv(nav_rewards), enabled=True)
    env.reset()
    training, bonuses = [], []
    for _ in nav_rewards:
        _, reward, _, _, info = env.step(0)
        training.append(reward)
        bonuses.append(info["positive_streak_bonus"])
    assert sum(training) == pytest.approx(sum(nav_rewards) + sum(bonuses))

