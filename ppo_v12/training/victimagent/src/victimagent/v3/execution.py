"""Causal V3 action-switch rules layered on the unchanged V1 portfolio ledger."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np


RULES = frozenset({
    "v3_control_daily_target", "v3_change_only", "v3_confirm_two_closes", "v3_min_hold_five"
})


class TradeDecisionEnv(gym.Wrapper):
    """Use close-t emitted IDs to accept an order for true open t+1.

    The base environment alone owns the market, ledger, events, NAV and reward.
    HOLD_SHARES calls its existing no-order path. All arms expose identical
    rule state, including the re-trained daily-target control.
    """

    def __init__(self, env: gym.Env, rule: str):
        if rule not in RULES:
            raise ValueError(f"unknown trade rule: {rule}")
        if not isinstance(env.action_space, spaces.Discrete) or env.action_space.n != 3:
            raise ValueError("V3 requires the frozen three target exposure IDs")
        if not isinstance(env.observation_space, spaces.Box) or len(env.observation_space.shape) != 1:
            raise ValueError("V3 requires a flat numeric base observation")
        super().__init__(env)
        self.rule = rule
        base = env.observation_space
        self.observation_space = spaces.Box(
            low=np.concatenate([base.low.flatten(), np.array([-1] * 3 + [0, -1])]),
            high=np.concatenate([base.high.flatten(), np.array([1] * 5)]),
            dtype=np.float32,
        )
        self.last_emitted: int | None = None
        self.last_executed: int | None = None
        self.pending: int | None = None
        self.pending_count = 0
        self.sessions_since_change: int | None = None

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        def action_code(value: int | None) -> float:
            return -1.0 if value is None else value / 2.0

        state = np.array([
            action_code(self.last_emitted), action_code(self.last_executed),
            action_code(self.pending), min(self.pending_count, 2) / 2.0,
            -1.0 if self.sessions_since_change is None else min(self.sessions_since_change, 6) / 6.0,
        ], dtype=np.float32)
        result = np.concatenate([np.asarray(obs, dtype=np.float32), state])
        if not self.observation_space.contains(result) or not np.isfinite(result).all():
            raise AssertionError("invalid V3 observation or rule state")
        return result

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.last_emitted = self.last_executed = self.pending = None
        self.pending_count = 0
        self.sessions_since_change = None
        obs, info = self.env.reset(seed=seed, options=options)
        return self._encode(obs), info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid V3 action: {action!r}")
        emitted = int(action)
        previous_executed = self.last_executed
        place_order = True
        reason = "daily_target"

        if previous_executed is None:
            reason = "initial_target"
            self.pending, self.pending_count = None, 0
        elif self.rule == "v3_change_only":
            place_order = emitted != self.last_emitted
            reason = "emitted_id_changed" if place_order else "same_emitted_id"
        elif self.rule == "v3_confirm_two_closes":
            if emitted == previous_executed:
                self.pending, self.pending_count = None, 0
                place_order, reason = False, "same_executed_id"
            else:
                if emitted == self.pending:
                    self.pending_count += 1
                else:
                    self.pending, self.pending_count = emitted, 1
                place_order = self.pending_count >= 2
                reason = "two_close_confirmation" if place_order else "awaiting_confirmation"
                if place_order:
                    self.pending, self.pending_count = None, 0
        elif self.rule == "v3_min_hold_five":
            if self.sessions_since_change is not None and self.sessions_since_change <= 5:
                place_order, reason = False, "five_session_lock"
            else:
                reason = "unlocked_daily_target"

        # This call remains the sole owner of prices and the ledger. A held order
        # never retroactively marks the unexecuted emitted target as owned shares.
        if place_order:
            obs, reward, terminated, truncated, info = self.env.step(emitted)
            self.last_executed = emitted
        else:
            obs, reward, terminated, truncated, info = self.env.step_target_weight(None)

        changed = place_order and previous_executed != emitted
        if changed:
            self.sessions_since_change = 1  # close after first executed open
        elif self.sessions_since_change is not None:
            self.sessions_since_change += 1
        self.last_emitted = emitted
        info = dict(info)
        info.update({
            "requested_action": emitted,
            "emitted_action": emitted,
            "executed_action": emitted if place_order else None,
            "order_placed": bool(place_order),
            "order_rejection_reason": None if place_order else reason,
            "trade_decision_rule": self.rule,
            "last_executed_action_id": self.last_executed,
        })
        return self._encode(obs), reward, terminated, truncated, info
