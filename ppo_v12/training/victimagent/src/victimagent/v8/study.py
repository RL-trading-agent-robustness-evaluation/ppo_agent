"""V8 final DQN victim: validate, adopt-or-fallback, final train, one-shot known period.

Stages (each writes exclusively-created records; nothing is overwritten):

1. ``preflight``            contract, committed code, data build hashes, fold audit
2. ``run_validation_fold``  one fold x five seeds x 500k steps (folds run in parallel)
3. ``aggregate``            preregistered adoption rule vs the frozen V5 parent record
4. ``run_final_seed``       adopted arm on its full pre-2022 window, one seed per call
5. ``known_period``         one-shot 2022-2025 confirmation of all five final seeds
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from victimagent.data.real_adapter import prepare_asset_bundle
from victimagent.data.splits import DateRange, SplitContract
from victimagent.dqn import parameter_sha256, seed_everything
from victimagent.evaluation import scaler_fingerprint, scaler_sha256
from victimagent.v2.folds import EPISODE_DECISIONS, audit_folds, load_rolling_folds
from victimagent.v2.registry import sha256_file
from victimagent.v2.run import build_v2_dqn
from victimagent.v3.execution import TradeDecisionEnv
from victimagent.v4.environment import SingleAssetTradingEnvV4
from victimagent.v4.evaluation import rollout
from victimagent.v5.study import _eval_model, _reserve
from victimagent.v8.features import V8_CONTEXT_FEATURES, prepare_v8_bundle

ARM = "v8_long_horizon"
REFERENCE = "v5_parent_five_session"
ACTION_WEIGHTS = (0.0, 0.5, 1.0)
RULE = "v3_min_hold_five"
FOLDS = ("fold_1", "fold_2", "fold_3")
SEEDS = (0, 1, 2, 3, 4)
BASELINES = ("cash", "buy_and_hold", "constant_50")
TIMESTEPS = 500_000
CUTOFF = pd.Timestamp("2022-01-01")
KNOWN = DateRange(pd.Timestamp("2022-01-01"), pd.Timestamp("2025-12-31"))
REPORTS = Path("reports/v8")
RECORDS = Path("experiments/v8_records")
ATTEMPTS = Path("experiments/v8_attempts")


# ----------------------------------------------------------------- contract
def contract(root: Path) -> dict:
    raw = yaml.safe_load((root / "configs/v8_final.yaml").read_text(encoding="utf-8"))
    fixed = {
        "version": 1, "status": "preregistered_before_v8_training", "asset": "SPY",
        "development_cutoff_exclusive": "2022-01-01", "fold_config": "configs/v8_folds.yaml",
        "candidate": ARM, "execution_rule": RULE, "environment": "SingleAssetTradingEnvV4",
        "reward": "log_true_close_nav_t_plus_1_over_true_close_nav_t",
        "transaction_cost_bps": 10, "episode_decisions": EPISODE_DECISIONS,
        "timesteps_per_run": TIMESTEPS, "known_period_accessed": False,
        "known_period_may_select_model": False, "initial_cash": 1000000.0,
    }
    for key, expected in fixed.items():
        if raw.get(key) != expected:
            raise ValueError(f"V8 fixed contract changed: {key}")
    if (tuple(raw["folds"]) != FOLDS or tuple(raw["seeds"]) != SEEDS
            or tuple(raw["final"]["seeds"]) != SEEDS
            or tuple(raw["action_weights"]) != ACTION_WEIGHTS
            or tuple(raw["baseline_policies"]) != BASELINES
            or tuple(raw["observation"]["market_features"]) + ("cash_rate_known",) != V8_CONTEXT_FEATURES
            or raw["adoption"]["reference_arm"] != REFERENCE):
        raise ValueError("V8 folds, seeds, actions, features or reference changed")
    return raw


def _git_commit(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def _clean(root: Path) -> None:
    out = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "src",
                          "scripts", "configs", "pyproject.toml"],
                         check=True, capture_output=True, text=True).stdout
    if out.strip():
        raise RuntimeError("V8 code/config must be committed before formal runs:\n" + out)


def _read(path: Path, *, before: pd.Timestamp | None, date_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if before is not None:
        frame = frame.loc[pd.to_datetime(frame[date_column]) < before].reset_index(drop=True)
    return frame


def _v8_inputs(root: Path, data_root: Path, *, before: pd.Timestamp | None = CUTOFF):
    cfg = contract(root)["data"]
    market = _read(data_root / cfg["processed_market"], before=before, date_column="date")
    events = _read(data_root / cfg["processed_events"], before=before, date_column="disexdt")
    return market, events


def _hp(root: Path) -> dict:
    stages = yaml.safe_load((root / "configs/dqn_v2_search.yaml").read_text(encoding="utf-8"))["stages"]
    stage = next(s for s in stages if s["id"] == "dqn_hyperparameters")
    return {**stage["fixed"], **stage["baseline"]}


def _packages() -> dict:
    result = {}
    for name in ("stable-baselines3", "torch", "numpy", "gymnasium", "pandas"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def preflight(root: Path, data_root: Path, *, require_clean: bool = True) -> dict:
    raw = contract(root)
    if require_clean:
        _clean(root)
    data = raw["data"]
    build_path = root / data["build_report"]
    if not build_path.exists():
        raise FileNotFoundError("run `python scripts/run_v8_final.py --build-data` first")
    build = json.loads(build_path.read_text(encoding="utf-8"))
    for rel, expected in build["processed_sha256"].items():
        if sha256_file(data_root / rel) != expected:
            raise ValueError(f"{rel} changed since the V8 data build")
    for name, expected in data["pinned_sha256"].items():
        if build["raw_sha256"].get(name) != expected:
            raise ValueError(f"V8 build used an unpinned {name}")
    market, events = _v8_inputs(root, data_root)
    if pd.to_datetime(market["date"]).max() >= CUTOFF:
        raise AssertionError("known-period rows survived development isolation")
    folds = load_rolling_folds(root / raw["fold_config"])
    audit_events = events.assign(disordinaryflg="Y")
    audit = audit_folds(folds, market, audit_events, ticker="SPY")
    if audit.status != "PASS":
        raise RuntimeError(f"V8 fold audit failed: {audit.to_dict()}")
    configs = {p: sha256_file(root / p) for p in (
        "configs/v8_final.yaml", "configs/v8_folds.yaml", "configs/dqn_v2_search.yaml")}
    return {"status": "PASS", "git_commit": _git_commit(root), "known_period_accessed": False,
            "config_sha256": configs, "data_build": {
                "raw_sha256": build["raw_sha256"], "processed_sha256": build["processed_sha256"]},
            "fold_audit": audit.to_dict(), "python": sys.version.split()[0],
            "packages": _packages()}


# ------------------------------------------------------------------ helpers
def _environment(bundle, *, training: bool = False, wrap: bool = True):
    env = SingleAssetTradingEnvV4(
        bundle.market_true, bundle.market_obs, bundle.observation_builder,
        initial_cash=1_000_000.0, transaction_cost_bps=10.0,
        episode_length=EPISODE_DECISIONS if training else None, random_start=training,
        dividend_events=bundle.dividend_events, split_events=bundle.split_events,
        action_weights=ACTION_WEIGHTS)
    return TradeDecisionEnv(env, RULE) if wrap else env


def _baselines(bundle) -> dict:
    result = {}
    for name in BASELINES:
        if name == "cash":
            choose = lambda _o, _i: 0  # noqa: E731
        elif name == "buy_and_hold":
            choose = lambda _o, i: 2 if i == 0 else None  # noqa: E731
        else:
            choose = lambda _o, _i: 1  # noqa: E731
        _rows, metrics, _err = rollout(_environment(bundle, wrap=False), choose, seed=SEEDS[0])
        result[name] = metrics
    return result


def _set_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(max(1, int(threads)))


def _train(root: Path, name: str, hp: dict, seed: int, train_bundle, timesteps: int):
    from victimagent.v3.run import _callback

    checkpoint = root / "artifacts/checkpoints/v8" / f"{name}.zip"
    if checkpoint.exists():
        raise FileExistsError(f"V8 checkpoint exists: {checkpoint}")
    seed_everything(seed)
    model = build_v2_dqn(_environment(train_bundle, training=True), hp, seed=seed)
    before = parameter_sha256(model)
    before_scaler = scaler_sha256(train_bundle.observation_builder)
    callback = _callback()
    started = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    seconds = time.time() - started
    if not callback.losses or not np.isfinite(callback.losses).all():
        raise RuntimeError("V8 training did not produce finite losses")
    if parameter_sha256(model) == before:
        raise RuntimeError("V8 model did not update")
    if scaler_sha256(train_bundle.observation_builder) != before_scaler:
        raise RuntimeError("V8 scaler mutated during training")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    return checkpoint, {"train_seconds": round(seconds, 1),
                        "final_logged_loss": float(callback.losses[-1])}


def _evaluate(root: Path, checkpoint: Path, bundle, seed: int, trajectory: Path) -> dict:
    from stable_baselines3 import DQN

    env = _environment(bundle)
    model = DQN.load(checkpoint, env=env, device="cpu")
    rows, measures, error, responsiveness, unchanged = _eval_model(
        model, env, bundle.observation_builder, seed)
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    if trajectory.exists():
        raise FileExistsError(f"V8 trajectory exists: {trajectory}")
    pd.DataFrame(rows).to_csv(trajectory, index=False)
    return {"technical_pass": bool(unchanged and abs(error) <= 1e-10),
            "state_responsive": bool(responsiveness["eligible"]),
            "responsiveness": responsiveness, "reward_nav_error": error,
            "evaluation_state_unchanged": unchanged,
            "trajectory_path": trajectory.relative_to(root).as_posix(),
            "trajectory_sha256": sha256_file(trajectory), **measures}


def _write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, default=str)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------- validation
def run_validation_fold(root: Path, data_root: Path, fold_id: str, *, threads: int = 1,
                        timesteps: int | None = None, seeds=SEEDS, tag: str = "") -> dict:
    timesteps = TIMESTEPS if timesteps is None else int(timesteps)
    if fold_id not in FOLDS:
        raise ValueError(f"unknown V8 fold: {fold_id}")
    formal = timesteps == TIMESTEPS and tuple(seeds) == SEEDS and not tag
    provenance = preflight(root, data_root, require_clean=formal)
    label = f"validation_{fold_id}{tag}"
    attempt, record_path = root / ATTEMPTS / f"{label}.json", root / RECORDS / f"{label}.json"
    if attempt.exists() or record_path.exists():
        raise FileExistsError(f"V8 {label} already attempted; inspect before retrying")
    _set_threads(threads)
    raw = contract(root)
    fold = next(f for f in load_rolling_folds(root / raw["fold_config"]).folds if f.fold_id == fold_id)
    market, events = _v8_inputs(root, data_root)
    train = prepare_v8_bundle(market, events, fit=fold.train, select=fold.train)
    val = prepare_v8_bundle(market, events, fit=fold.train, select=fold.validation)
    hp = _hp(root)
    _reserve(attempt, {"status": "reserved", "fold": fold_id, "created_at_utc": _now(), **provenance})
    results, status, error = [], "FAIL", None
    baselines = _baselines(val)
    try:
        for seed in seeds:
            name = f"{ARM}_{fold_id}_seed{seed}{tag}"
            print(f"[{_now()}] V8 {fold_id} seed {seed}: training {timesteps} steps "
                  f"({train.metadata['first_date']}..{train.metadata['last_date']})", flush=True)
            checkpoint, train_info = _train(root, name, hp, seed, train, timesteps)
            evaluation = _evaluate(root, checkpoint, val, seed,
                                   root / "artifacts/trajectories/v8" / f"{name}_validation.csv")
            bh = baselines["buy_and_hold"]["final_wealth"]
            row = {"fold": fold_id, "seed": seed, "known_period_accessed": False,
                   "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                   "checkpoint_sha256": sha256_file(checkpoint), **train_info, **evaluation,
                   "log_relative_wealth_vs_buy_and_hold": math.log(evaluation["final_wealth"] / bh)}
            results.append(row)
            print(f"[{_now()}] V8 {fold_id} seed {seed}: wealth {row['final_wealth']:.4f} vs "
                  f"B&H {bh:.4f}; exposure {row['average_exposure']:.2f}; "
                  f"pass={row['technical_pass']}; {train_info['train_seconds']}s", flush=True)
        status = "PASS" if all(r["technical_pass"] for r in results) else "FAIL"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write_new(record_path, {
            "status": status, "error": error, "arm": ARM, "fold": fold_id,
            "formal": formal, "seeds": list(seeds), "timesteps_per_run": timesteps,
            "created_at_utc": _now(), "known_period_accessed": False,
            "preflight": provenance, "bundle": {"train": train.metadata, "validation": val.metadata},
            "scaler": scaler_fingerprint(train.observation_builder),
            "hyperparameters": hp, "baseline_metrics": baselines, "per_seed": results})
    return json.loads(record_path.read_text(encoding="utf-8"))


def _fold_records(root: Path) -> dict:
    records = {}
    for fold in FOLDS:
        record = json.loads((root / RECORDS / f"validation_{fold}.json").read_text(encoding="utf-8"))
        if (record["status"] != "PASS" or not record["formal"] or record["fold"] != fold
                or record["seeds"] != list(SEEDS) or record["timesteps_per_run"] != TIMESTEPS
                or record["known_period_accessed"] is not False):
            raise ValueError(f"V8 {fold} record is not a complete formal PASS")
        if sorted(r["seed"] for r in record["per_seed"]) != list(SEEDS):
            raise ValueError(f"V8 {fold} missing or duplicate seeds")
        for row in record["per_seed"]:
            for kind in ("checkpoint", "trajectory"):
                if sha256_file(root / row[f"{kind}_path"]) != row[f"{kind}_sha256"]:
                    raise ValueError(f"V8 {fold} seed {row['seed']} {kind} hash mismatch")
        records[fold] = record
    commits = {r["preflight"]["git_commit"] for r in records.values()}
    builds = {json.dumps(r["preflight"]["data_build"], sort_keys=True) for r in records.values()}
    if len(commits) != 1 or len(builds) != 1:
        raise ValueError("V8 fold records come from different commits or data builds")
    return records


def decide(v8: dict, reference: dict) -> dict:
    """Pure adoption rule; inputs are {fold: {seed: final_wealth}}."""
    per_fold = {}
    for fold in FOLDS:
        diffs = [math.log(v8[fold][s] / reference[fold][s]) for s in SEEDS]
        per_fold[fold] = {"median_log_wealth_difference": float(np.median(diffs)),
                          "seed_wins": int(sum(d > 0 for d in diffs)),
                          "per_seed_log_wealth_difference": diffs}
    folds_won = sum(v["median_log_wealth_difference"] > 0 for v in per_fold.values())
    return {"per_fold": per_fold, "folds_won": int(folds_won),
            "adopted_arm": ARM if folds_won >= 2 else REFERENCE}


def aggregate(root: Path) -> dict:
    raw = contract(root)
    records = _fold_records(root)
    reference = json.loads((root / raw["adoption"]["reference_record"]).read_text(encoding="utf-8"))
    if (reference.get("arm") != REFERENCE or reference.get("status") != "PASS"
            or reference.get("known_period_accessed") is not False):
        raise ValueError("V5 parent reference record is missing or not PASS")
    ref_rows = {(r["fold"], r["seed"]): r for r in reference["per_fold_seed"]}
    if set(ref_rows) != {(f, s) for f in FOLDS for s in SEEDS}:
        raise ValueError("V5 parent reference record is incomplete")
    same_window = {}
    for fold in FOLDS:
        mine = records[fold]["baseline_metrics"]["buy_and_hold"]
        theirs = reference["baseline_metrics"][fold]["buy_and_hold"]
        same_window[fold] = {"v8": mine["final_wealth"], "v5": theirs["final_wealth"],
                             "start": [mine["start_date"], theirs["start_date"]],
                             "end": [mine["end_date"], theirs["end_date"]]}
        if (mine["start_date"], mine["end_date"]) != (theirs["start_date"], theirs["end_date"]) or \
                not math.isclose(mine["final_wealth"], theirs["final_wealth"], rel_tol=1e-9):
            raise ValueError(f"{fold}: V8 and V5 buy-and-hold windows differ: {same_window[fold]}")
    v8 = {f: {r["seed"]: r["final_wealth"] for r in records[f]["per_seed"]} for f in FOLDS}
    ref = {f: {s: ref_rows[(f, s)]["final_wealth"] for s in SEEDS} for f in FOLDS}
    decision = decide(v8, ref)
    all_pass = all(r["technical_pass"] for rec in records.values() for r in rec["per_seed"])
    if not all_pass:
        decision["adopted_arm"] = REFERENCE
    objective = {}
    for fold in FOLDS:
        rel = [r["log_relative_wealth_vs_buy_and_hold"] for r in records[fold]["per_seed"]]
        bh = records[fold]["baseline_metrics"]["buy_and_hold"]["final_wealth"]
        ref_rel = [math.log(ref[fold][s] / bh) for s in SEEDS]
        objective[fold] = {"v8_median_log_relative_wealth": float(np.median(rel)),
                           "v8_wins_vs_buy_and_hold": int(sum(x > 0 for x in rel)),
                           "v5_median_log_relative_wealth": float(np.median(ref_rel)),
                           "v5_wins_vs_buy_and_hold": int(sum(x > 0 for x in ref_rel))}
    summary = {
        "status": "PASS", "created_at_utc": _now(), "known_period_accessed": False,
        "git_commit": _git_commit(root),
        "v8_record_commit": records[FOLDS[0]]["preflight"]["git_commit"],
        "rule": raw["adoption"]["rule"], "all_v8_runs_technical_pass": all_pass,
        **decision, "same_window_buy_and_hold": same_window,
        "v6_objective_report_only": objective,
        "v8_runs": {f: [{k: r[k] for k in ("seed", "final_wealth", "cagr", "sharpe_excess_cash",
                                            "max_drawdown", "average_exposure", "total_turnover",
                                            "state_responsive", "log_relative_wealth_vs_buy_and_hold")}
                        for r in records[f]["per_seed"]] for f in FOLDS},
        "baselines": {f: records[f]["baseline_metrics"] for f in FOLDS},
    }
    _write_new(root / REPORTS / "VALIDATION_DECISION.json", summary)
    lines = [
        "# V8 validation decision (pre-2022 only)", "",
        f"Adopted final arm: **{summary['adopted_arm']}** "
        f"(V8 beat the V5 parent's paired median log wealth in {summary['folds_won']}/3 folds; "
        "rule: adopt V8 if >= 2/3 and every V8 run technically passes).", "",
        "| Fold | Median log(V8/V5) | Seed wins | V8 median log(W/B&H) | V8 wins vs B&H | "
        "V5 median log(W/B&H) | V5 wins vs B&H |", "|---|---:|---:|---:|---:|---:|---:|"]
    for fold in FOLDS:
        p, o = summary["per_fold"][fold], objective[fold]
        lines.append(f"| {fold} | {p['median_log_wealth_difference']:+.4f} | {p['seed_wins']}/5 | "
                     f"{o['v8_median_log_relative_wealth']:+.4f} | {o['v8_wins_vs_buy_and_hold']}/5 | "
                     f"{o['v5_median_log_relative_wealth']:+.4f} | {o['v5_wins_vs_buy_and_hold']}/5 |")
    lines += ["", "Known period (2022-2025) not read. Every run is retained in "
              "`experiments/v8_records/`.", ""]
    with (root / REPORTS / "VALIDATION_DECISION.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return summary


# -------------------------------------------------------------------- final
def _final_window(raw: dict, arm: str) -> DateRange:
    spec = raw["final"][arm]
    return DateRange(pd.Timestamp(spec["train_start"]), pd.Timestamp(spec["train_end"]))


def _v5_contract(train: DateRange) -> SplitContract:
    contract_ = SplitContract(train=train, validation=KNOWN,
                              test=DateRange(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")),
                              eligible_start=pd.Timestamp("2009-01-01"),
                              eligible_end=pd.Timestamp("2026-12-31"))
    contract_.validate()
    return contract_


def _final_bundles(root: Path, data_root: Path, arm: str, *, known: bool):
    """Train bundle (and optionally the 2022-2025 bundle) for the adopted arm."""
    raw = contract(root)
    window = _final_window(raw, arm)
    before = None if known else CUTOFF
    if arm == ARM:
        market, events = _v8_inputs(root, data_root, before=before)
        train = prepare_v8_bundle(market, events, fit=window, select=window)
        test = prepare_v8_bundle(market, events, fit=window, select=KNOWN) if known else None
        return train, test
    spec = raw["final"][REFERENCE]
    market = _read(data_root / spec["processed_market"], before=before, date_column="date")
    events = _read(data_root / spec["processed_events"], before=before, date_column="disexdt")
    split = _v5_contract(window)
    train = prepare_asset_bundle(market, events, split, ticker="SPY", partition="train")
    test = (prepare_asset_bundle(market, events, split, ticker="SPY", partition="validation")
            if known else None)
    return train, test


def _decision(root: Path) -> dict:
    return json.loads((root / REPORTS / "VALIDATION_DECISION.json").read_text(encoding="utf-8"))


def run_final_seed(root: Path, data_root: Path, seed: int, *, threads: int = 1) -> dict:
    if seed not in SEEDS:
        raise ValueError("final seeds are fixed to 0-4")
    raw = contract(root)
    decision = _decision(root)
    arm = decision["adopted_arm"]
    provenance = preflight(root, data_root)
    label = f"final_{arm}_seed{seed}"
    attempt, record_path = root / ATTEMPTS / f"{label}.json", root / RECORDS / f"{label}.json"
    if attempt.exists() or record_path.exists():
        raise FileExistsError(f"V8 {label} already attempted")
    _set_threads(threads)
    train, _ = _final_bundles(root, data_root, arm, known=False)
    timesteps = int(raw["final"][arm]["timesteps"])
    _reserve(attempt, {"status": "reserved", "arm": arm, "seed": seed, **provenance})
    status, error, info, checkpoint = "FAIL", None, {}, None
    try:
        print(f"[{_now()}] V8 final {arm} seed {seed}: training {timesteps} steps "
              f"({train.metadata['first_date']}..{train.metadata['last_date']})", flush=True)
        checkpoint, info = _train(root, f"final_{arm}_seed{seed}", _hp(root), seed, train, timesteps)
        status = "PASS"
        print(f"[{_now()}] V8 final {arm} seed {seed}: done in {info['train_seconds']}s", flush=True)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write_new(record_path, {
            "status": status, "error": error, "arm": arm, "seed": seed, "timesteps": timesteps,
            "created_at_utc": _now(), "known_period_accessed": False, "preflight": provenance,
            "decision_sha256": sha256_file(root / REPORTS / "VALIDATION_DECISION.json"),
            "train_window": train.metadata, "scaler": scaler_fingerprint(train.observation_builder),
            "action_weights": list(ACTION_WEIGHTS), "execution_rule": RULE,
            "checkpoint_path": checkpoint.relative_to(root).as_posix() if checkpoint else None,
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None, **info})
    return json.loads(record_path.read_text(encoding="utf-8"))


def known_period(root: Path, data_root: Path) -> dict:
    """One-shot 2022-2025 confirmation. It cannot change the frozen victim."""
    raw = contract(root)
    _clean(root)
    arm = _decision(root)["adopted_arm"]
    finals = []
    for seed in SEEDS:
        record = json.loads((root / RECORDS / f"final_{arm}_seed{seed}.json").read_text(encoding="utf-8"))
        if record["status"] != "PASS" or sha256_file(root / record["checkpoint_path"]) != record["checkpoint_sha256"]:
            raise ValueError(f"final seed {seed} is not a verified PASS")
        finals.append(record)
    if len({r["preflight"]["git_commit"] for r in finals}) != 1:
        raise ValueError("final seeds come from different commits")
    attempt = root / ATTEMPTS / "known_period.json"
    _reserve(attempt, {"status": "reserved", "arm": arm, "created_at_utc": _now()})
    train, test = _final_bundles(root, data_root, arm, known=True)
    if scaler_fingerprint(train.observation_builder) != finals[0]["scaler"]:
        raise ValueError("known-period scaler differs from the final training scaler")
    baselines = _baselines(test)
    bh = baselines["buy_and_hold"]["final_wealth"]
    runs = []
    for record in finals:
        seed = record["seed"]
        result = _evaluate(root, root / record["checkpoint_path"], test, seed,
                           root / "artifacts/trajectories/v8" / f"known_period_{arm}_seed{seed}.csv")
        result.update(seed=seed, log_relative_wealth_vs_buy_and_hold=math.log(result["final_wealth"] / bh))
        runs.append(result)
    summary = {
        "status": "PASS", "label": raw["known_period"]["label"], "arm": arm,
        "created_at_utc": _now(), "git_commit": _git_commit(root),
        "window": test.metadata, "may_change_final_model": False,
        "baselines": baselines, "per_seed": runs,
        "median_final_wealth": float(np.median([r["final_wealth"] for r in runs])),
        "median_log_relative_wealth_vs_buy_and_hold":
            float(np.median([r["log_relative_wealth_vs_buy_and_hold"] for r in runs])),
        "wins_vs_buy_and_hold": int(sum(r["log_relative_wealth_vs_buy_and_hold"] > 0 for r in runs)),
    }
    _write_new(root / REPORTS / "KNOWN_PERIOD_CONFIRMATION.json", summary)
    manifest = {
        "victim": "dqn_final", "arm": arm, "git_commit": finals[0]["preflight"]["git_commit"],
        "environment": "victimagent.v4.environment.SingleAssetTradingEnvV4",
        "wrapper": f"victimagent.v3.execution.TradeDecisionEnv(rule='{RULE}')",
        "action_weights": list(ACTION_WEIGHTS),
        "observation_context_features": finals[0]["scaler"]["columns"],
        "scaler": finals[0]["scaler"], "train_window": finals[0]["train_window"],
        "checkpoints": [{"seed": r["seed"], "path": r["checkpoint_path"],
                         "sha256": r["checkpoint_sha256"]} for r in finals],
        "attack_protocol": "evaluate every attack on all five seeds; no seed selection",
    }
    _write_new(root / REPORTS / "FINAL_VICTIM_MANIFEST.json", manifest)
    lines = [f"# V8 final victim — known-period confirmation ({raw['known_period']['label']})", "",
             f"Arm: **{arm}**. Window {test.metadata['first_date']}..{test.metadata['last_date']}. "
             "Run once after the victim was frozen; it cannot change the model.", "",
             "| Policy | Final wealth | CAGR | Sharpe vs cash | Max DD | Exposure | log(W/B&H) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in baselines.items():
        lines.append(f"| {name} | {m['final_wealth']:.3f} | {m['cagr']:.2%} | "
                     f"{m['sharpe_excess_cash'] if m['sharpe_excess_cash'] is None else round(m['sharpe_excess_cash'], 3)} | "
                     f"{m['max_drawdown']:.2%} | {m['average_exposure']:.2f} | "
                     f"{math.log(m['final_wealth'] / bh):+.4f} |")
    for r in runs:
        lines.append(f"| DQN seed {r['seed']} | {r['final_wealth']:.3f} | {r['cagr']:.2%} | "
                     f"{r['sharpe_excess_cash'] if r['sharpe_excess_cash'] is None else round(r['sharpe_excess_cash'], 3)} | "
                     f"{r['max_drawdown']:.2%} | {r['average_exposure']:.2f} | "
                     f"{r['log_relative_wealth_vs_buy_and_hold']:+.4f} |")
    lines += ["", f"Wins vs buy-and-hold: {summary['wins_vs_buy_and_hold']}/5.", ""]
    with (root / REPORTS / "KNOWN_PERIOD_CONFIRMATION.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return summary
