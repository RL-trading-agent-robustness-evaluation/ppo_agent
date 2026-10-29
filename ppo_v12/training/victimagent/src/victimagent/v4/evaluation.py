"""V4 close-to-close evaluation; never reads a later open or model label."""

from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pandas as pd


def rollout(env, choose, *, seed: int = 0, q_values=None):
    """Evaluate a deterministic policy; None from choose means hold shares."""
    observation, reset_info = env.reset(seed=seed)
    rows = []
    initial_nav = float(reset_info["initial_close_nav"])
    while True:
        index = len(rows)
        chosen = choose(observation, index)
        q = q_values(observation) if q_values is not None else None
        if q is not None and (np.asarray(q).ndim != 1 or not np.isfinite(q).all()):
            raise AssertionError("V4 evaluation Q values are not finite")
        if chosen is None:
            next_observation, reward, terminated, truncated, info = env.step_target_weight(None)
        else:
            next_observation, reward, terminated, truncated, info = env.step(int(chosen))
        if not env.observation_space.contains(next_observation):
            raise AssertionError("V4 observation outside space")
        if rows and not math.isclose(float(info["start_close_nav"]), rows[-1]["end_nav"],
                                     rel_tol=1e-12, abs_tol=1e-6):
            raise AssertionError("V4 NAV path does not telescope")
        rows.append({
            "decision_date": str(info["decision_date"].date()),
            "execution_date": str(info["execution_date"].date()),
            "valuation_date": str(info["valuation_date"].date()),
            "reward_available_date": str(info["reward_available_date"].date()),
            "next_observation_date": str(info["next_observation_date"].date()),
            "action": -1 if chosen is None else int(chosen),
            "executed_action": info.get("executed_action"),
            "order_placed": info.get("order_placed", chosen is not None),
            "true_execution_open": float(info["true_execution_open"]),
            "true_valuation_close": float(info["true_valuation_close"]),
            "start_nav": float(info["start_close_nav"]),
            "end_nav": float(info["end_close_nav"]),
            "net_return": float(info["net_return"]),
            "gross_return": float(info["gross_return"]),
            "cash_benchmark_return": float(info["cash_benchmark_return"]),
            "transaction_cost": float(info["transaction_cost"]),
            "trade_notional": float(info["trade"]["trade_notional"]),
            "turnover": float(info["turnover"]),
            "realized_weight": float(info["realized_weight"]),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            **({f"q_{i}": float(v) for i, v in enumerate(q)} if q is not None else {}),
        })
        observation = next_observation
        if terminated or truncated:
            break
    if not rows[-1]["terminated"] or rows[-1]["truncated"]:
        raise AssertionError("V4 full evaluation must end at the data boundary")
    error = float(sum(row["reward"] for row in rows) -
                  math.log(rows[-1]["end_nav"] / initial_nav))
    if abs(error) > 1e-10:
        raise AssertionError("V4 reward–NAV invariant failed")
    return rows, metrics(rows), error


def metrics(rows):
    if not rows:
        raise ValueError("V4 trajectory cannot be empty")
    net = np.asarray([r["net_return"] for r in rows], dtype=float)
    cash = np.asarray([r["cash_benchmark_return"] for r in rows], dtype=float)
    nav = np.asarray([rows[0]["start_nav"]] + [r["end_nav"] for r in rows], dtype=float)
    if not np.isfinite(nav).all() or (nav <= 0).any() or not np.isfinite(net).all():
        raise AssertionError("V4 NAV and returns must be finite and positive")
    if any(r["reward_available_date"] != r["next_observation_date"] or
           r["valuation_date"] != r["next_observation_date"] for r in rows):
        raise AssertionError("V4 reward/observation availability mismatch")
    excess = net - cash
    std = float(excess.std(ddof=1)) if len(rows) > 1 else 0.0
    elapsed = (pd.Timestamp(rows[-1]["valuation_date"]) -
               pd.Timestamp(rows[0]["decision_date"])).days / 365.25
    if elapsed <= 0:
        raise AssertionError("V4 evaluation needs positive calendar duration")
    actions = Counter(str(r["action"]) for r in rows)
    return {
        "steps": len(rows),
        "start_date": rows[0]["decision_date"],
        "end_date": rows[-1]["valuation_date"],
        "final_wealth": float(nav[-1] / nav[0]),
        "cagr": float((nav[-1] / nav[0]) ** (1 / elapsed) - 1),
        "annualized_mean_excess_return": float(excess.mean() * 252),
        "annualized_mean_gross_excess_return": float(np.mean(
            np.asarray([r["gross_return"] for r in rows]) - cash) * 252),
        "sharpe_excess_cash": float(excess.mean() / std * np.sqrt(252)) if std > 1e-12 else None,
        "annualized_volatility": float(net.std(ddof=1) * np.sqrt(252)) if len(rows) > 1 else 0.0,
        "max_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1)),
        "average_exposure": float(np.mean([r["realized_weight"] for r in rows])),
        "total_turnover": float(sum(r["turnover"] for r in rows)),
        "number_of_trades": sum(abs(r["trade_notional"]) > 1e-8 for r in rows),
        "total_transaction_cost": float(sum(r["transaction_cost"] for r in rows)),
        "action_distribution": {a: count / len(rows) for a, count in sorted(actions.items())},
    }
