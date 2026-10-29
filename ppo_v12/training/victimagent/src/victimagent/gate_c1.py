"""Gate C1 structural diagnostics and deterministic checkpoint selection."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from victimagent.features import CONTEXT_FEATURES


@dataclass(frozen=True)
class ResponsivenessDiagnostics:
    steps: int
    unique_actions: int
    action_distribution: dict[str, float]
    normalized_action_entropy: float
    q_values_vary_across_states: bool
    q_range_max: float
    cyclic_context_action_changes: int
    cyclic_context_disagreement_rate: float
    neutral_context_action_changes: int
    neutral_context_disagreement_rate: float
    nonconstant_policy: bool
    market_context_changes_actions: bool
    eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_q_values(model: Any, observations: np.ndarray) -> np.ndarray:
    """Return finite DQN Q-values for a fixed offline observation matrix."""

    import torch

    matrix = np.asarray(observations, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError("responsiveness audit requires at least two observations")
    with torch.no_grad():
        tensor = torch.as_tensor(matrix, dtype=torch.float32, device=model.device)
        values = model.q_net(tensor).detach().cpu().numpy()
    if values.ndim != 2 or not np.isfinite(values).all():
        raise AssertionError("C1 requires a finite two-dimensional Q-value matrix")
    return values


def diagnose_responsiveness(model: Any, observations: np.ndarray) -> ResponsivenessDiagnostics:
    """Test whether decisions respond to market context on fixed validation states.

    Account-state dimensions stay paired with their original rows. Only the
    seven frozen context dimensions are changed: once by a deterministic
    half-period cyclic permutation and once by replacing them with zero, which
    is the train-fit standardized mean. These are offline decision diagnostics;
    they do not create an alternative P&L path and never use test data.
    """

    matrix = np.asarray(observations, dtype=np.float32)
    context_width = len(CONTEXT_FEATURES)
    if matrix.ndim != 2 or matrix.shape[1] <= context_width:
        raise ValueError("observation matrix violates Observation Contract V1")

    original_q = model_q_values(model, matrix)
    original_actions = np.argmax(original_q, axis=1)

    cyclic = matrix.copy()
    shift = max(1, len(matrix) // 2)
    cyclic[:, :context_width] = np.roll(matrix[:, :context_width], shift, axis=0)
    cyclic_actions = np.argmax(model_q_values(model, cyclic), axis=1)

    neutral = matrix.copy()
    neutral[:, :context_width] = 0.0
    neutral_actions = np.argmax(model_q_values(model, neutral), axis=1)

    counts = Counter(int(x) for x in original_actions)
    steps = len(original_actions)
    distribution = {str(k): counts.get(k, 0) / steps for k in range(original_q.shape[1])}
    probabilities = np.asarray([v for v in distribution.values() if v > 0], dtype=float)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = entropy / float(np.log(original_q.shape[1]))

    q_ranges = np.ptp(original_q, axis=0)
    q_values_vary = bool(np.any(q_ranges != 0.0))
    cyclic_changes = int(np.count_nonzero(original_actions != cyclic_actions))
    neutral_changes = int(np.count_nonzero(original_actions != neutral_actions))
    nonconstant = len(counts) >= 2
    market_sensitive = cyclic_changes > 0 or neutral_changes > 0

    return ResponsivenessDiagnostics(
        steps=steps,
        unique_actions=len(counts),
        action_distribution=distribution,
        normalized_action_entropy=normalized_entropy,
        q_values_vary_across_states=q_values_vary,
        q_range_max=float(np.max(q_ranges)),
        cyclic_context_action_changes=cyclic_changes,
        cyclic_context_disagreement_rate=cyclic_changes / steps,
        neutral_context_action_changes=neutral_changes,
        neutral_context_disagreement_rate=neutral_changes / steps,
        nonconstant_policy=nonconstant,
        market_context_changes_actions=market_sensitive,
        eligible=nonconstant and q_values_vary and market_sensitive,
    )


def select_checkpoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select among eligible seeds using the frozen validation metric.

    Highest validation Sharpe after costs wins. Exact ties are resolved by
    lower turnover and then lower seed. Ineligible seeds remain in the report
    but cannot be selected.
    """

    eligible = [row for row in rows if row.get("eligible") and row.get("sharpe") is not None]
    if not eligible:
        raise ValueError("Gate C1 has no eligible checkpoint")
    return min(
        eligible,
        key=lambda row: (-float(row["sharpe"]), float(row["turnover"]), int(row["seed"])),
    )
