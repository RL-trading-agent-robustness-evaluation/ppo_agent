from __future__ import annotations

import gymnasium as gym


class PositiveStreakRewardWrapper(gym.Wrapper):
    """Add a capped, proportional bonus to consecutive positive NAV rewards.

    The wrapped environment remains the only owner of prices, actions, fills,
    observations, ledger state and true NAV reward.  Set ``enabled=False`` for
    the matched control so both arms expose the same diagnostic fields.
    """

    def __init__(self, env: gym.Env, *, beta: float = 0.10,
                 streak_cap: int = 5, enabled: bool = True):
        super().__init__(env)
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if streak_cap < 1:
            raise ValueError("streak_cap must be at least one")
        self.beta = float(beta)
        self.streak_cap = int(streak_cap)
        self.enabled = bool(enabled)
        self.positive_return_streak = 0

    def reset(self, **kwargs):
        self.positive_return_streak = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        nav_reward = float(reward)
        if nav_reward > 0.0:
            self.positive_return_streak += 1
        else:
            self.positive_return_streak = 0
        multiplier_steps = max(min(self.positive_return_streak, self.streak_cap) - 1, 0)
        bonus = (self.beta * multiplier_steps * max(nav_reward, 0.0)
                 if self.enabled else 0.0)
        training_reward = nav_reward + bonus
        diagnostics = dict(info)
        diagnostics.update({
            "nav_log_reward": nav_reward,
            "positive_return_streak": self.positive_return_streak,
            "positive_streak_bonus": bonus,
            "training_reward": training_reward,
            "streak_reward_enabled": self.enabled,
        })
        return observation, training_reward, terminated, truncated, diagnostics

