"""Deterministic clean baselines for the single-asset environment."""

from __future__ import annotations

from collections import Counter
import math

import numpy as np

from victimagent.data.real_adapter import RealDataBundle
from victimagent.environment import ACTION_WEIGHTS, SingleAssetTradingEnvV1


BASELINE_NAMES = ("cash", "buy_and_hold", "constant_50", "ma_20_60", "random")


def run_baseline(
    bundle: RealDataBundle,
    name: str,
    *,
    transaction_cost_bps: float = 10.0,
    seed: int = 0,
    include_nav_curve: bool = False,
) -> dict[str, object]:
    """Run one full-period baseline and return auditable performance metrics."""

    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline: {name}")
    if name == "ma_20_60" and "ma_fast_above_slow" not in bundle.market_obs:
        raise KeyError("MA baseline requires ma_fast_above_slow")

    env = SingleAssetTradingEnvV1(
        bundle.market_true,
        bundle.market_obs,
        bundle.observation_builder,
        initial_cash=1_000_000.0,
        transaction_cost_bps=transaction_cost_bps,
        dividend_events=bundle.dividend_events,
        split_events=bundle.split_events,
    )
    _, reset_info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    decision_index = int(reset_info["decision_index"])
    rewards: list[float] = []
    net_returns: list[float] = []
    cash_returns: list[float] = []
    nav_path: list[float] = []
    turnovers: list[float] = []
    exposures: list[float] = []
    valuation_dates: list[str] = []
    total_cost = 0.0
    number_of_trades = 0
    actions: Counter[str] = Counter()
    first_nav: float | None = None
    first_execution_date = None
    last_nav: float | None = None
    last_valuation_date = None
    step_number = 0

    while True:
        if name == "cash":
            action, hold = 0, False
        elif name == "constant_50":
            action, hold = 1, False
        elif name == "buy_and_hold":
            action, hold = (2, False) if step_number == 0 else (None, True)
        elif name == "ma_20_60":
            signal = int(bundle.market_obs.loc[decision_index, "ma_fast_above_slow"])
            action, hold = (2 if signal else 0), False
        else:
            action, hold = int(rng.integers(0, len(ACTION_WEIGHTS))), False

        if hold:
            _, reward, terminated, truncated, info = env.step_target_weight(None)
            action_label = "HOLD_SHARES"
        else:
            assert action is not None
            _, reward, terminated, truncated, info = env.step(int(action))
            action_label = str(int(action))
        if truncated:
            raise AssertionError("clean baselines must never truncate")

        if first_nav is None:
            first_nav = float(info["start_pretrade_nav"])
            first_execution_date = info["execution_date"]
            nav_path.append(first_nav)
        last_nav = float(info["end_pretrade_nav"])
        last_valuation_date = info["valuation_date"]
        nav_path.append(last_nav)
        valuation_dates.append(str(info["valuation_date"].date()))
        rewards.append(float(reward))
        net_returns.append(float(info["net_return"]))
        cash_returns.append(float(info["cash_benchmark_return"]))
        turnovers.append(float(info["turnover"]))
        exposures.append(float(info["realized_weight"]))
        total_cost += float(info["transaction_cost"])
        if abs(float(info["trade"]["trade_notional"])) > 1e-8:
            number_of_trades += 1
        actions[action_label] += 1
        decision_index += 1
        step_number += 1
        if terminated:
            break

    assert first_nav is not None and last_nav is not None
    reward_sum = float(np.sum(rewards))
    nav_log_change = math.log(last_nav / first_nav)
    if not math.isclose(reward_sum, nav_log_change, rel_tol=0.0, abs_tol=1e-10):
        raise AssertionError("reward sum does not reconstruct final NAV")

    elapsed_years = (last_valuation_date - first_execution_date).days / 365.25
    if elapsed_years <= 0:
        raise AssertionError("baseline period must span positive calendar time")
    total_return = last_nav / first_nav - 1.0
    cagr = (last_nav / first_nav) ** (1.0 / elapsed_years) - 1.0
    returns = np.asarray(net_returns, dtype=float)
    excess = returns - np.asarray(cash_returns, dtype=float)
    excess_std = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    sharpe = (
        float(np.mean(excess) / excess_std * np.sqrt(252.0))
        if excess_std > 1e-12
        else None
    )
    mean_daily_excess_return = float(np.mean(excess))
    annualized_mean_excess_return = mean_daily_excess_return * 252.0
    annualized_volatility = (
        float(np.std(returns, ddof=1) * np.sqrt(252.0))
        if len(returns) > 1
        else 0.0
    )
    nav = np.asarray(nav_path, dtype=float)
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    action_counts = dict(sorted(actions.items()))
    action_distribution = {
        label: count / len(rewards) for label, count in action_counts.items()
    }
    step_abs_returns = np.abs(returns)
    max_return_index = int(np.argmax(step_abs_returns))

    result: dict[str, object] = {
        "name": name,
        "seed": seed if name == "random" else None,
        "steps": len(rewards),
        "start_date": str(first_execution_date.date()),
        "end_date": str(last_valuation_date.date()),
        "initial_nav": first_nav,
        "final_nav": last_nav,
        "final_wealth": last_nav / first_nav,
        "total_return": total_return,
        "cagr": cagr,
        "mean_daily_excess_return": mean_daily_excess_return,
        "annualized_mean_excess_return": annualized_mean_excess_return,
        "sharpe_excess_cash": sharpe,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": float(np.min(drawdown)),
        "total_turnover": float(np.sum(turnovers)),
        "mean_daily_turnover": float(np.mean(turnovers)),
        "total_transaction_cost": total_cost,
        "average_exposure": float(np.mean(exposures)),
        "number_of_trades": number_of_trades,
        "action_counts": action_counts,
        "action_distribution": action_distribution,
        "max_abs_daily_return": float(step_abs_returns[max_return_index]),
        "max_abs_daily_return_date": valuation_dates[max_return_index],
        "nav_finite_positive": bool(np.isfinite(nav).all() and (nav > 0).all()),
        "reward_nav_invariant_error": reward_sum - nav_log_change,
    }
    if include_nav_curve:
        result["nav_curve"] = {
            "dates": [str(first_execution_date.date()), *valuation_dates],
            "normalized_wealth": (nav / first_nav).tolist(),
        }
    return result


def run_baseline_suite(
    bundle: RealDataBundle,
    *,
    random_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    transaction_cost_bps: float = 10.0,
    include_nav_curve: bool = False,
) -> list[dict[str, object]]:
    results = [
        run_baseline(
            bundle,
            name,
            transaction_cost_bps=transaction_cost_bps,
            include_nav_curve=include_nav_curve,
        )
        for name in ("cash", "buy_and_hold", "constant_50", "ma_20_60")
    ]
    results.extend(
        run_baseline(
            bundle,
            "random",
            transaction_cost_bps=transaction_cost_bps,
            seed=seed,
            include_nav_curve=include_nav_curve,
        )
        for seed in random_seeds
    )
    return results
