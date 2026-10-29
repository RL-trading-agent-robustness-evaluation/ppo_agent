"""V9: three preregistered arms on top of V8 (gate, relative reward, both).

Stages mirror V8 and reuse its data, contract checks and helpers:

1. ``run_validation(arm, fold)``  one arm x one fold x five seeds x 500k steps
2. ``aggregate``                  preregistered selection vs the V8 incumbent
3. ``run_final_seed``             winner on 1994-02-01..2021-12-31 (if a V9 arm won)
4. ``known_period``               one-shot 2022-2025 confirmation + victim manifest
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from victimagent.data.splits import DateRange
from victimagent.dqn import parameter_sha256, seed_everything
from victimagent.evaluation import scaler_fingerprint, scaler_sha256
from victimagent.v2.folds import load_rolling_folds
from victimagent.v2.registry import sha256_file
from victimagent.v2.run import build_v2_dqn
from victimagent.v4.evaluation import rollout
from victimagent.v5.study import _eval_model, _reserve
from victimagent.v8 import study as v8
from victimagent.v8.features import build_v8_features, prepare_v8_bundle
from victimagent.v9.wrappers import (
    REGIME_COLUMN, RelativeRewardWrapper, TrendGateWrapper, spy_log_returns,
)

ARMS = {
    "v9_relative_reward": {"gate": False, "relative_reward": True},
    "v9_trend_gate": {"gate": True, "relative_reward": False},
    "v9_gate_relative_reward": {"gate": True, "relative_reward": True},
}
FOLDS, SEEDS, TIMESTEPS = v8.FOLDS, v8.SEEDS, v8.TIMESTEPS
REPORTS = Path("reports/v9")
RECORDS = Path("experiments/v9_records")
ATTEMPTS = Path("experiments/v9_attempts")
FINAL_WINDOW = DateRange(pd.Timestamp("1994-02-01"), pd.Timestamp("2021-12-31"))


def contract(root: Path) -> dict:
    raw = yaml.safe_load((root / "configs/v9_final.yaml").read_text(encoding="utf-8"))
    fixed = {"version": 1, "status": "preregistered_before_v9_training",
             "parent": "v8_final", "timesteps_per_run": TIMESTEPS,
             "known_period_accessed": False, "known_period_may_select_model": False}
    for key, expected in fixed.items():
        if raw.get(key) != expected:
            raise ValueError(f"V9 fixed contract changed: {key}")
    if raw["arms"] != ARMS or tuple(raw["folds"]) != FOLDS or tuple(raw["seeds"]) != SEEDS:
        raise ValueError("V9 arms, folds or seeds changed")
    v8.contract(root)  # data, features, env, actions and rule are inherited unchanged
    return raw


# ----------------------------------------------------------------- bundles
def v9_bundle(market: pd.DataFrame, events: pd.DataFrame, *, fit: DateRange,
              select: DateRange):
    """V8 bundle plus the raw trend flag (observed stream) and SPY log returns."""
    bundle = prepare_v8_bundle(market, events, fit=fit, select=select)
    features = build_v8_features(market)
    flags = features.set_index("date")
    dates = pd.to_datetime(bundle.market_obs["date"])
    above = (flags["feature_price"] > flags["feature_sma_200"]).reindex(dates)
    if above.isna().any():
        raise ValueError("trend flag missing for selected sessions")
    market_obs = bundle.market_obs.copy()
    market_obs[REGIME_COLUMN] = above.to_numpy().astype(np.int8)
    cash = flags["cash_distribution"].reindex(dates).to_numpy()
    spy = spy_log_returns(bundle.market_true, cash)
    metadata = {**bundle.metadata,
                "trend_up_share": float(market_obs[REGIME_COLUMN].mean())}
    return replace(bundle, market_obs=market_obs, metadata=metadata), spy


def _environment(bundle, spy, arm: str, *, training: bool):
    spec = ARMS[arm]
    env = v8._environment(bundle, training=training)          # V4 + min-hold rule
    if spec["gate"]:
        env = TrendGateWrapper(env)
    if spec["relative_reward"] and training:                    # evaluation: true NAV reward
        env = RelativeRewardWrapper(env, spy)
    return env


# ------------------------------------------------------------ train / eval
def _train(root, name, hp, seed, bundle, spy, arm, timesteps):
    from victimagent.v3.run import _callback

    checkpoint = root / "artifacts/checkpoints/v9" / f"{name}.zip"
    if checkpoint.exists():
        raise FileExistsError(f"V9 checkpoint exists: {checkpoint}")
    seed_everything(seed)
    model = build_v2_dqn(_environment(bundle, spy, arm, training=True), hp, seed=seed)
    before, scaler = parameter_sha256(model), scaler_sha256(bundle.observation_builder)
    callback = _callback()
    started = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    if not callback.losses or not np.isfinite(callback.losses).all():
        raise RuntimeError("V9 training did not produce finite losses")
    if parameter_sha256(model) == before or scaler_sha256(bundle.observation_builder) != scaler:
        raise RuntimeError("V9 model did not update or scaler mutated")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    return checkpoint, {"train_seconds": round(time.time() - started, 1)}


def _action_trace(rows: list[dict], bundle, arm: str) -> dict:
    """Separate the policy request, trend gate request, and actual held-order action."""
    flags = dict(zip(pd.to_datetime(bundle.market_obs["date"]).dt.date.astype(str),
                     bundle.market_obs[REGIME_COLUMN].astype(bool)))
    requested, gated, executed = [], [], []
    overrides = 0
    for row in rows:
        request = int(row["action"])
        forced = bool(ARMS[arm]["gate"] and flags[row["decision_date"]])
        target = 2 if forced else request
        row.update(agent_action=request, gate_forced_full=forced, gated_action=target)
        requested.append(str(request))
        gated.append(str(target))
        executed.append(str(row["executed_action"]))
        overrides += int(target != request)

    def distribution(actions):
        return {key: actions.count(key) / len(actions) for key in sorted(set(actions))}

    return {"agent_action_distribution": distribution(requested),
            "gated_action_distribution": distribution(gated),
            "executed_action_distribution": distribution(executed),
            "gate_override_share": overrides / len(rows),
            "gate_forced_share": sum(row["gate_forced_full"] for row in rows) / len(rows)}


def _baselines_with_rule(bundle, spy):
    baselines = v8._baselines(bundle)
    # A deterministic trend-only control: full above SMA-200 and cash below,
    # executed with the same five-session rule, ledger and transaction costs.
    env = _environment(bundle, spy, "v9_trend_gate", training=False)
    rows, measures, error = rollout(env, lambda _obs, _i: 0, seed=SEEDS[0])
    if abs(error) > 1e-10:
        raise AssertionError("V9 trend-only rule reward–NAV invariant failed")
    baselines["sma_200_gate_only"] = {**measures,
        "action_trace": _action_trace(rows, bundle, "v9_trend_gate")}
    return baselines


def _evaluate(root, checkpoint, bundle, spy, arm, seed, trajectory):
    from stable_baselines3 import DQN

    env = _environment(bundle, spy, arm, training=False)
    model = DQN.load(checkpoint, env=env, device="cpu")
    rows, measures, error, responsiveness, unchanged = _eval_model(
        model, env, bundle.observation_builder, seed)
    action_trace = _action_trace(rows, bundle, arm)
    if trajectory.exists():
        raise FileExistsError(f"V9 trajectory exists: {trajectory}")
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(trajectory, index=False)
    return {**action_trace, "technical_pass": bool(unchanged and abs(error) <= 1e-10),
            "state_responsive": bool(responsiveness["eligible"]),
            "responsiveness": responsiveness, "reward_nav_error": error,
            "trajectory_path": trajectory.relative_to(root).as_posix(),
            "trajectory_sha256": sha256_file(trajectory), **measures}


# -------------------------------------------------------------- validation
def run_validation(root: Path, data_root: Path, arm: str, fold_id: str, *, threads: int = 1,
                   timesteps: int | None = None, seeds=SEEDS, tag: str = "") -> dict:
    if arm not in ARMS or fold_id not in FOLDS:
        raise ValueError(f"unknown V9 arm/fold: {arm}/{fold_id}")
    timesteps = TIMESTEPS if timesteps is None else int(timesteps)
    formal = timesteps == TIMESTEPS and tuple(seeds) == SEEDS and not tag
    contract(root)
    provenance = {**v8.preflight(root, data_root, require_clean=formal),
                  "v9_config_sha256": sha256_file(root / "configs/v9_final.yaml")}
    label = f"{arm}_{fold_id}{tag}"
    attempt, record_path = root / ATTEMPTS / f"{label}.json", root / RECORDS / f"{label}.json"
    if attempt.exists() or record_path.exists():
        raise FileExistsError(f"V9 {label} already attempted; inspect before retrying")
    v8._set_threads(threads)
    fold = next(f for f in load_rolling_folds(root / "configs/v8_folds.yaml").folds
                if f.fold_id == fold_id)
    market, events = v8._v8_inputs(root, data_root)
    train, train_spy = v9_bundle(market, events, fit=fold.train, select=fold.train)
    val, val_spy = v9_bundle(market, events, fit=fold.train, select=fold.validation)
    hp = v8._hp(root)
    _reserve(attempt, {"status": "reserved", "arm": arm, "fold": fold_id, **provenance})
    baselines = _baselines_with_rule(val, val_spy)
    bh = baselines["buy_and_hold"]["final_wealth"]
    results, status, error = [], "FAIL", None
    try:
        for seed in seeds:
            name = f"{label}_seed{seed}"
            print(f"[{v8._now()}] V9 {arm} {fold_id} seed {seed}: training {timesteps} steps",
                  flush=True)
            checkpoint, info = _train(root, name, hp, seed, train, train_spy, arm, timesteps)
            evaluation = _evaluate(root, checkpoint, val, val_spy, arm, seed,
                                   root / "artifacts/trajectories/v9" / f"{name}_validation.csv")
            row = {"fold": fold_id, "seed": seed, **info, **evaluation,
                   "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                   "checkpoint_sha256": sha256_file(checkpoint),
                   "log_relative_wealth_vs_buy_and_hold": math.log(evaluation["final_wealth"] / bh)}
            results.append(row)
            print(f"[{v8._now()}] V9 {arm} {fold_id} seed {seed}: wealth "
                  f"{row['final_wealth']:.4f} vs B&H {bh:.4f}; exposure "
                  f"{row['average_exposure']:.2f}; {info['train_seconds']}s", flush=True)
        status = "PASS" if all(r["technical_pass"] for r in results) else "FAIL"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        v8._write_new(record_path, {
            "status": status, "error": error, "arm": arm, "arm_spec": ARMS[arm],
            "fold": fold_id, "formal": formal, "seeds": list(seeds),
            "timesteps_per_run": timesteps, "created_at_utc": v8._now(),
            "known_period_accessed": False, "preflight": provenance,
            "bundle": {"train": train.metadata, "validation": val.metadata},
            "baseline_metrics": baselines, "per_seed": results})
    return json.loads(record_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------- selection
def _incumbent(root: Path) -> tuple[str, dict]:
    """Per fold/seed validation wealth of the arm V8 adopted."""
    decision = json.loads((root / "reports/v8/VALIDATION_DECISION.json").read_text(encoding="utf-8"))
    arm = decision["adopted_arm"]
    if arm == v8.ARM:
        records = v8._fold_records(root)
        wealth = {f: {r["seed"]: r["final_wealth"] for r in records[f]["per_seed"]} for f in FOLDS}
    else:
        ref = json.loads((root / v8.contract(root)["adoption"]["reference_record"])
                         .read_text(encoding="utf-8"))
        wealth = {f: {r["seed"]: r["final_wealth"] for r in ref["per_fold_seed"] if r["fold"] == f}
                  for f in FOLDS}
    return arm, wealth


def select(arms: dict, bh: dict, incumbent: dict) -> dict:
    """Pure preregistered rule. arms: {arm: {fold: {seed: wealth}}}; bh: {fold: wealth}."""
    scores = {}
    for arm, wealth in arms.items():
        per_fold = {f: float(np.median([math.log(wealth[f][s] / bh[f]) for s in SEEDS]))
                    for f in FOLDS}
        paired = {f: float(np.median([math.log(wealth[f][s] / incumbent[f][s]) for s in SEEDS]))
                  for f in FOLDS}
        scores[arm] = {"score_mean_of_fold_median_log_wealth_vs_bh": float(np.mean(list(per_fold.values()))),
                       "per_fold_median_log_wealth_vs_bh": per_fold,
                       "per_fold_median_log_wealth_vs_incumbent": paired,
                       "folds_beating_incumbent": int(sum(v > 0 for v in paired.values()))}
    best = max(scores, key=lambda a: scores[a]["score_mean_of_fold_median_log_wealth_vs_bh"])
    adopted = best if scores[best]["folds_beating_incumbent"] >= 2 else None
    return {"scores": scores, "best_v9_arm": best, "adopted_v9_arm": adopted}


def aggregate(root: Path) -> dict:
    raw = contract(root)
    records = {}
    for arm in ARMS:
        for fold in FOLDS:
            record = json.loads((root / RECORDS / f"{arm}_{fold}.json").read_text(encoding="utf-8"))
            if (record.get("status") != "PASS" or record.get("formal") is not True
                    or record.get("arm") != arm or record.get("arm_spec") != ARMS[arm]
                    or record.get("fold") != fold or record.get("seeds") != list(SEEDS)
                    or record.get("timesteps_per_run") != TIMESTEPS
                    or record.get("known_period_accessed") is not False
                    or sorted(r["seed"] for r in record.get("per_seed", [])) != list(SEEDS)):
                raise ValueError(f"V9 {arm} {fold} is not a complete formal PASS")
            if any(r.get("fold") != fold or r.get("technical_pass") is not True
                   or not isinstance(r.get("state_responsive"), bool)
                   or not math.isfinite(r.get("final_wealth", float("nan")))
                   or r["final_wealth"] <= 0 for r in record["per_seed"]):
                raise ValueError(f"V9 {arm} {fold} has invalid per-seed metrics")
            if "sma_200_gate_only" not in record.get("baseline_metrics", {}):
                raise ValueError(f"V9 {arm} {fold} lacks the trend-only rule control")
            for row in record["per_seed"]:
                for kind in ("checkpoint", "trajectory"):
                    if sha256_file(root / row[f"{kind}_path"]) != row[f"{kind}_sha256"]:
                        raise ValueError(f"V9 {arm} {fold} seed {row['seed']} {kind} hash mismatch")
            records[(arm, fold)] = record
    commits = {r["preflight"]["git_commit"] for r in records.values()}
    v9_hashes = {r["preflight"].get("v9_config_sha256") for r in records.values()}
    v8_hashes = {json.dumps(r["preflight"]["config_sha256"], sort_keys=True)
                 for r in records.values()}
    builds = {json.dumps(r["preflight"]["data_build"], sort_keys=True)
              for r in records.values()}
    if (commits != {v8._git_commit(root)}
            or v9_hashes != {sha256_file(root / "configs/v9_final.yaml")}
            or len(v8_hashes) != 1 or len(builds) != 1):
        raise ValueError("V9 records have mixed or stale commit/config/data provenance")
    incumbent_records = v8._fold_records(root)
    for fold in FOLDS:
        old = incumbent_records[fold]["baseline_metrics"]["buy_and_hold"]
        new = records[(next(iter(ARMS)), fold)]["baseline_metrics"]["buy_and_hold"]
        if ((old["start_date"], old["end_date"]) != (new["start_date"], new["end_date"])
                or not math.isclose(old["final_wealth"], new["final_wealth"], rel_tol=1e-9)):
            raise ValueError(f"V9 {fold} buy-and-hold differs from the incumbent window")
    bh = {}
    for fold in FOLDS:
        values = {r["baseline_metrics"]["buy_and_hold"]["final_wealth"]
                  for (a, f), r in records.items() if f == fold}
        if len(values) != 1:
            raise ValueError(f"V9 arms disagree on {fold} buy-and-hold")
        bh[fold] = values.pop()
        rules = {r["baseline_metrics"]["sma_200_gate_only"]["final_wealth"]
                 for (a, f), r in records.items() if f == fold}
        if len(rules) != 1:
            raise ValueError(f"V9 arms disagree on {fold} trend-only rule")
    incumbent_arm, incumbent = _incumbent(root)
    wealth = {arm: {f: {r["seed"]: r["final_wealth"] for r in records[(arm, f)]["per_seed"]}
                    for f in FOLDS} for arm in ARMS}
    decision = select(wealth, bh, incumbent)
    final_arm = decision["adopted_v9_arm"] or incumbent_arm
    summary = {"status": "PASS", "created_at_utc": v8._now(), "known_period_accessed": False,
               "rule": raw["selection"]["rule"], "incumbent_arm": incumbent_arm,
               "final_arm": final_arm, **decision, "buy_and_hold": bh,
               "sma_200_gate_only": {f: records[(next(iter(ARMS)), f)]
                                          ["baseline_metrics"]["sma_200_gate_only"]
                                     for f in FOLDS},
               "runs": {arm: {f: [{k: r[k] for k in ("seed", "final_wealth", "sharpe_excess_cash",
                                                     "max_drawdown", "average_exposure",
                                                     "total_turnover", "state_responsive",
                                                     "agent_action_distribution",
                                                     "gated_action_distribution",
                                                     "executed_action_distribution",
                                                     "gate_override_share", "gate_forced_share")}
                                   for r in records[(arm, f)]["per_seed"]] for f in FOLDS}
                        for arm in ARMS}}
    v8._write_new(root / REPORTS / "VALIDATION_DECISION.json", summary)
    lines = ["# V9 validation decision (pre-2022 only)", "",
             f"Incumbent (from V8): **{incumbent_arm}**. Best V9 arm: **{decision['best_v9_arm']}**. "
             f"Final victim arm: **{final_arm}**.", "",
             "| Arm | Score | fold_1 vs B&H | fold_2 vs B&H | fold_3 vs B&H | Folds beating incumbent |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, s in decision["scores"].items():
        f = s["per_fold_median_log_wealth_vs_bh"]
        lines.append(f"| {arm} | {s['score_mean_of_fold_median_log_wealth_vs_bh']:+.4f} | "
                     f"{f['fold_1']:+.4f} | {f['fold_2']:+.4f} | {f['fold_3']:+.4f} | "
                     f"{s['folds_beating_incumbent']}/3 |")
    lines += ["", "Values are median over seeds of log(wealth / same-window buy-and-hold). "
              "Known period not read.", "",
              "| Fold | Buy-and-hold wealth | SMA-200 gate-only wealth |",
              "|---|---:|---:|"]
    for fold in FOLDS:
        lines.append(f"| {fold} | {bh[fold]:.4f} | "
                     f"{summary['sma_200_gate_only'][fold]['final_wealth']:.4f} |")
    lines += ["", "Per-seed requested, gated and executed action distributions, "
              "plus forced/overridden shares, are in VALIDATION_DECISION.json.", ""]
    with (root / REPORTS / "VALIDATION_DECISION.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return summary


# ------------------------------------------------------------ final stages
def _decision(root: Path) -> dict:
    return json.loads((root / REPORTS / "VALIDATION_DECISION.json").read_text(encoding="utf-8"))


def run_final_seed(root: Path, data_root: Path, seed: int, *, threads: int = 1) -> dict:
    contract(root)
    arm = _decision(root)["adopted_v9_arm"]
    if arm is None:
        raise RuntimeError("No V9 arm was adopted; the V8 final victim stands (nothing to train)")
    provenance = {**v8.preflight(root, data_root),
                  "v9_config_sha256": sha256_file(root / "configs/v9_final.yaml")}
    label = f"final_{arm}_seed{seed}"
    attempt, record_path = root / ATTEMPTS / f"{label}.json", root / RECORDS / f"{label}.json"
    if attempt.exists() or record_path.exists():
        raise FileExistsError(f"V9 {label} already attempted")
    v8._set_threads(threads)
    market, events = v8._v8_inputs(root, data_root)
    bundle, spy = v9_bundle(market, events, fit=FINAL_WINDOW, select=FINAL_WINDOW)
    _reserve(attempt, {"status": "reserved", "arm": arm, "seed": seed, **provenance})
    status, error, info, checkpoint = "FAIL", None, {}, None
    try:
        print(f"[{v8._now()}] V9 final {arm} seed {seed}: training {TIMESTEPS} steps", flush=True)
        checkpoint, info = _train(root, label, v8._hp(root), seed, bundle, spy, arm, TIMESTEPS)
        status = "PASS"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        v8._write_new(record_path, {
            "status": status, "error": error, "arm": arm, "arm_spec": ARMS[arm], "seed": seed,
            "timesteps": TIMESTEPS, "created_at_utc": v8._now(), "preflight": provenance,
            "known_period_accessed": False, "train_window": bundle.metadata,
            "scaler": scaler_fingerprint(bundle.observation_builder),
            "checkpoint_path": checkpoint.relative_to(root).as_posix() if checkpoint else None,
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None, **info})
    return json.loads(record_path.read_text(encoding="utf-8"))


def known_period(root: Path, data_root: Path) -> dict:
    contract(root)
    provenance = v8.preflight(root, data_root)  # recheck the full processed-data build
    decision_path = root / REPORTS / "VALIDATION_DECISION.json"
    decision = _decision(root)
    if decision.get("status") != "PASS" or decision.get("known_period_accessed") is not False:
        raise ValueError("V9 validation decision is not a pre-2022 PASS")
    arm = decision["adopted_v9_arm"]
    if arm is None:
        raise RuntimeError("No V9 arm was adopted; use the V8 known-period report")
    finals = []
    for seed in SEEDS:
        record = json.loads((root / RECORDS / f"final_{arm}_seed{seed}.json").read_text(encoding="utf-8"))
        saved = record.get("preflight", {})
        if (record.get("status") != "PASS" or record.get("arm") != arm
                or record.get("seed") != seed or record.get("timesteps") != TIMESTEPS
                or record.get("known_period_accessed") is not False
                or saved.get("git_commit") != provenance["git_commit"]
                or saved.get("config_sha256") != provenance["config_sha256"]
                or saved.get("data_build") != provenance["data_build"]
                or saved.get("v9_config_sha256") != sha256_file(root / "configs/v9_final.yaml")
                or sha256_file(root / record["checkpoint_path"]) != record["checkpoint_sha256"]):
            raise ValueError(f"V9 final seed {seed} is not a verified same-version PASS")
        finals.append(record)
    _reserve(root / ATTEMPTS / "known_period.json",
             {"status": "reserved", "arm": arm, "decision_sha256": sha256_file(decision_path),
              "git_commit": provenance["git_commit"],
              "checkpoints_sha256": {str(r["seed"]): r["checkpoint_sha256"] for r in finals}})
    market, events = v8._v8_inputs(root, data_root, before=None)
    test, spy = v9_bundle(market, events, fit=FINAL_WINDOW, select=v8.KNOWN)
    if scaler_fingerprint(test.observation_builder) != finals[0]["scaler"]:
        raise ValueError("known-period scaler differs from the final training scaler")
    baselines = _baselines_with_rule(test, spy)
    bh = baselines["buy_and_hold"]["final_wealth"]
    runs = []
    for record in finals:
        result = _evaluate(root, root / record["checkpoint_path"], test, spy, arm, record["seed"],
                           root / "artifacts/trajectories/v9" / f"known_period_{arm}_seed{record['seed']}.csv")
        result.update(seed=record["seed"],
                      log_relative_wealth_vs_buy_and_hold=math.log(result["final_wealth"] / bh))
        runs.append(result)
    summary = {"status": "PASS", "label": "known_period_confirmation_not_untouched_test",
               "arm": arm, "window": test.metadata, "baselines": baselines, "per_seed": runs,
               "wins_vs_buy_and_hold": int(sum(r["log_relative_wealth_vs_buy_and_hold"] > 0 for r in runs)),
               "median_log_relative_wealth_vs_buy_and_hold":
                   float(np.median([r["log_relative_wealth_vs_buy_and_hold"] for r in runs]))}
    v8._write_new(root / REPORTS / "KNOWN_PERIOD_CONFIRMATION.json", summary)
    v8._write_new(root / REPORTS / "FINAL_VICTIM_MANIFEST.json", {
        "victim": "dqn_final", "arm": arm, "arm_spec": ARMS[arm],
        "git_commit": finals[0]["preflight"]["git_commit"],
        "environment": "victimagent.v4.environment.SingleAssetTradingEnvV4",
        "wrappers": ["victimagent.v3.execution.TradeDecisionEnv(rule='v3_min_hold_five')"]
                    + (["victimagent.v9.wrappers.TrendGateWrapper"] if ARMS[arm]["gate"] else []),
        "evaluation_reward": "true log NAV (relative reward is training-only)",
        "scaler": finals[0]["scaler"],
        "checkpoints": [{"seed": r["seed"], "path": r["checkpoint_path"],
                         "sha256": r["checkpoint_sha256"]} for r in finals],
        "attack_protocol": "evaluate every attack on all five seeds; no seed selection"})
    lines = [f"# V9 final victim — known-period confirmation ({summary['label']})", "",
             f"Arm **{arm}**, {test.metadata['first_date']}..{test.metadata['last_date']}.", "",
             "| Policy | Final wealth | CAGR | Max DD | Exposure | log(W/B&H) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, m in baselines.items():
        lines.append(f"| {name} | {m['final_wealth']:.3f} | {m['cagr']:.2%} | {m['max_drawdown']:.2%} | "
                     f"{m['average_exposure']:.2f} | {math.log(m['final_wealth'] / bh):+.4f} |")
    for r in runs:
        lines.append(f"| DQN seed {r['seed']} | {r['final_wealth']:.3f} | {r['cagr']:.2%} | "
                     f"{r['max_drawdown']:.2%} | {r['average_exposure']:.2f} | "
                     f"{r['log_relative_wealth_vs_buy_and_hold']:+.4f} |")
    with (root / REPORTS / "KNOWN_PERIOD_CONFIRMATION.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return summary
