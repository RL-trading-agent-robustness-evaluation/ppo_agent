"""Exact log-NAV gap accounting for paired V4-clock trajectories.

The up/down partition is descriptive, not a causal estimate of timing skill:
the agent's exposure, dividends, cash interest, and prices jointly determine it.
"""

from __future__ import annotations

import math


def _close(a: float, b: float, *, rel_tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-8)


def decompose(agent: list[dict], buy_hold: list[dict]) -> dict:
    if not agent or len(agent) != len(buy_hold):
        raise ValueError("Paired trajectories need the same nonzero number of rows")
    sums = {"up_day_gross_gap": 0.0, "down_day_gross_gap": 0.0,
            "flat_day_gross_gap": 0.0, "agent_cost_drag": 0.0,
            "buy_hold_entry_cost_credit": 0.0}
    counts = {"up_days": 0, "down_days": 0, "flat_days": 0}
    weights = []
    for index, (a, b) in enumerate(zip(agent, buy_hold)):
        if (a["decision_date"], a["valuation_date"]) != (b["decision_date"], b["valuation_date"]):
            raise ValueError(f"Decision/valuation date mismatch at row {index}")
        for name, row in (("agent", a), ("buy_hold", b)):
            start, end = float(row["start_nav"]), float(row["end_nav"])
            gross, net = float(row["gross_return"]), float(row["net_return"])
            cost = float(row["transaction_cost"])
            if not all(math.isfinite(x) for x in (start, end, gross, net, cost)) or start <= 0 or end <= 0 or cost < 0 or min(gross, net) <= -1:
                raise ValueError(f"Invalid {name} NAV/return/cost at row {index}")
            if not _close(end / start - 1, net) or not _close(gross - net, cost / start):
                raise ValueError(f"{name} ledger mismatch at row {index}")
        if index:
            if not _close(float(a["start_nav"]), float(agent[index - 1]["end_nav"])) or not _close(float(b["start_nav"]), float(buy_hold[index - 1]["end_nav"])):
                raise ValueError(f"NAV discontinuity at row {index}")
        gross_gap = math.log1p(float(a["gross_return"])) - math.log1p(float(b["gross_return"]))
        market = float(b["gross_return"])
        if market > 0:
            kind = "up"
        elif market < 0:
            kind = "down"
        else:
            kind = "flat"
        counts[f"{kind}_days"] += 1
        sums[f"{kind}_day_gross_gap"] += gross_gap
        sums["agent_cost_drag"] += math.log1p(float(a["net_return"])) - math.log1p(float(a["gross_return"]))
        sums["buy_hold_entry_cost_credit"] += math.log1p(float(b["gross_return"])) - math.log1p(float(b["net_return"]))
        weights.append(float(a["realized_weight"]))
    net_gap = sum(sums.values())
    expected = math.log(float(agent[-1]["end_nav"]) / float(agent[0]["start_nav"])) - math.log(float(buy_hold[-1]["end_nav"]) / float(buy_hold[0]["start_nav"]))
    if not math.isclose(net_gap, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("Relative reward–NAV invariant failed")
    scale = 252.0 / len(agent)
    return {"steps": len(agent), **counts,
            "net_log_wealth_gap": net_gap,
            "annualized_log_wealth_gap": net_gap * scale,
            "annualized_components": {name: value * scale for name, value in sums.items()},
            "agent_final_wealth": float(agent[-1]["end_nav"]) / float(agent[0]["start_nav"]),
            "buy_hold_final_wealth": float(buy_hold[-1]["end_nav"]) / float(buy_hold[0]["start_nav"]),
            "average_agent_exposure": sum(weights) / len(weights)}
