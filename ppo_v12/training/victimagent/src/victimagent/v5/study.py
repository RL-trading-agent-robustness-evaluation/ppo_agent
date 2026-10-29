"""Pre-2022 paired V5 exposure-floor study on the frozen V4 clock."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from victimagent.data.real_adapter import prepare_asset_bundle
from victimagent.dqn import parameter_sha256, seed_everything
from victimagent.evaluation import optimizer_sha256, scaler_sha256
from victimagent.v2.folds import EPISODE_DECISIONS, audit_folds, load_rolling_folds
from victimagent.v2.registry import sha256_file
from victimagent.v2.run import _responsiveness_diagnostics, build_v2_dqn
from victimagent.v3.execution import TradeDecisionEnv
from victimagent.v4.environment import SingleAssetTradingEnvV4
from victimagent.v4.evaluation import rollout


ARMS = ("v5_parent_five_session", "v5_floor_50_five_session")
ACTION_WEIGHTS = {
    "v5_parent_five_session": (0., .5, 1.),
    "v5_floor_50_five_session": (.5, .75, 1.),
}
RULE = "v3_min_hold_five"
BASELINES = ("cash", "buy_and_hold", "constant_50")
FOLDS = ("fold_1", "fold_2", "fold_3")
SEEDS = (0, 1, 2, 3, 4)


def _read_development(path: Path, *, date_column: str | None = None) -> pd.DataFrame:
    """Discard forbidden rows chunk by chunk before feature/bundle construction."""
    selected = []
    for chunk in pd.read_csv(path, chunksize=50000):
        column = date_column or next((c for c in ("disexdt", "exdt") if c in chunk), None)
        if column is None or column not in chunk:
            raise KeyError("V5 input has no development date column")
        dates = pd.to_datetime(chunk[column], errors="raise")
        selected.append(chunk.loc[dates < pd.Timestamp("2022-01-01")].copy())
    if not selected:
        raise ValueError("empty V5 input")
    return pd.concat(selected, ignore_index=True)


def _market_events(data_root: Path):
    market = _read_development(data_root / "data/processed/market_base_with_splits.csv",
                               date_column="date")
    events = _read_development(data_root / "data/interim/distribution_events.csv")
    return market, events


def contract(root: Path) -> dict:
    raw = yaml.safe_load((root / "configs/v5_exposure_floor.yaml").read_text(encoding="utf-8"))
    required = {
        "version": 1, "status": "preregistered_before_v5_training", "asset": "SPY",
        "development_cutoff_exclusive": "2022-01-01", "fold_config": "configs/v2_rolling_folds.yaml",
        "reward": "log_true_close_nav_t_plus_1_over_true_close_nav_t",
        "transaction_cost_bps": 10, "episode_decisions": 504,
        "timesteps_per_fold_seed": 200000, "known_period_accessed": False,
        "known_period_may_select_model": False,
        "execution_rule": RULE,
        "parent_hyperparameters": "configs/dqn_v2_search.yaml",
        "observation": "frozen_v3_full_10d_plus_five_rule_state_inputs",
        "decision": "close_t", "execution": "true_raw_open_t_plus_1",
        "valuation_and_next_observation": "close_t_plus_1",
        "initial_cash": 1000000.0, "train_random_start": True,
        "evaluation_deterministic": True, "evaluation_state_frozen": True,
        "feature_expansion": "deferred_separate_version",
        "gaussian_observation_noise": "deferred_separate_ablation_train_only",
    }
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise ValueError(f"V5 fixed contract changed: {key}")
    if tuple(raw["folds"]) != FOLDS or tuple(raw["seeds"]) != SEEDS or tuple(raw["arms"]) != ARMS:
        raise ValueError("V5 paired folds, seeds or arms changed")
    if tuple(raw["baseline_policies"]) != BASELINES or raw["action_weights"] != {
        arm: list(weights) for arm, weights in ACTION_WEIGHTS.items()
    }:
        raise ValueError("V5 controls or action set changed")
    return raw


def _git_commit(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


def _clean(root: Path) -> None:
    out = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--",
                          "src", "scripts", "configs", "pyproject.toml"],
                         check=True, capture_output=True, text=True).stdout
    if out.strip():
        raise RuntimeError("V5 research code/config must be committed before formal training")


def preflight(root: Path, data_root: Path) -> dict:
    """Hash full inputs as bytes; parse only pre-2022 market/events."""
    raw = contract(root)
    _clean(root)
    paths = {
        "v5_config": root / "configs/v5_exposure_floor.yaml",
        "v4_config": root / "configs/v4_close_mark.yaml",
        "v2_hp_config": root / "configs/dqn_v2_search.yaml",
        "fold_config": root / raw["fold_config"],
        "market": data_root / "data/processed/market_base_with_splits.csv",
        "distributions": data_root / "data/interim/distribution_events.csv",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    market, events = _market_events(data_root)
    folds = load_rolling_folds(paths["fold_config"])
    if tuple(f.fold_id for f in folds.folds) != FOLDS:
        raise ValueError("V5 fold IDs do not match contract")
    result = audit_folds(folds, market, events, ticker="SPY")
    if result.status != "PASS":
        raise RuntimeError(f"V5 pre-2022 fold audit failed: {result.to_dict()}")
    packages = {}
    for name in ("stable-baselines3", "torch", "numpy", "gymnasium", "pandas"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "status": "PASS", "git_commit": _git_commit(root),
        "known_period_accessed": False,
        "hashes": hashes, "fold_audit": result.to_dict(),
        "python": sys.version.split()[0], "packages": packages,
    }


def _inputs(root: Path, data_root: Path):
    market, events = _market_events(data_root)
    folds = load_rolling_folds(root / "configs/v2_rolling_folds.yaml")
    hp_stage = next(stage for stage in yaml.safe_load(
        (root / "configs/dqn_v2_search.yaml").read_text(encoding="utf-8"))["stages"]
        if stage["id"] == "dqn_hyperparameters")
    hp = {**hp_stage["fixed"], **hp_stage["baseline"]}
    return market, events, folds, hp


def _environment(bundle, *, arm: str | None = None, training: bool = False):
    if arm is not None and arm not in ARMS:
        raise ValueError(f"unregistered V5 arm: {arm}")
    env = SingleAssetTradingEnvV4(
        bundle.market_true, bundle.market_obs, bundle.observation_builder,
        initial_cash=1_000_000., transaction_cost_bps=10.,
        episode_length=EPISODE_DECISIONS if training else None,
        random_start=training, dividend_events=bundle.dividend_events,
        split_events=bundle.split_events,
        action_weights=ACTION_WEIGHTS[arm] if arm is not None else ACTION_WEIGHTS[ARMS[0]],
    )
    return TradeDecisionEnv(env, RULE) if arm is not None else env


def _eval_model(model, env, builder, seed):
    import torch

    model.policy.set_training_mode(False)
    before = (parameter_sha256(model), optimizer_sha256(model),
              scaler_sha256(builder),
              model.replay_buffer.size() if model.replay_buffer is not None else 0)

    def predict(obs, _):
        action, _state = model.predict(obs, deterministic=True)
        return int(np.asarray(action).item())

    def q_values(obs):
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).reshape(1, -1)
            return model.q_net(tensor).detach().cpu().numpy()[0]

    rows, measures, error = rollout(env, predict, seed=seed, q_values=q_values)
    responsiveness = _responsiveness_diagnostics(
        model, env, seed=seed, context_width=len(builder.context_features))
    after = (parameter_sha256(model), optimizer_sha256(model),
             scaler_sha256(builder),
             model.replay_buffer.size() if model.replay_buffer is not None else 0)
    return rows, measures, error, responsiveness, before == after


def _train_one(root, arm, hp, fold, seed, train_bundle, val_bundle, *, timesteps):
    from stable_baselines3 import DQN
    from victimagent.v3.run import _callback

    name = f"v5_{arm}_{fold}_seed{seed}"
    checkpoint = root / "artifacts/checkpoints/v5" / f"{name}.zip"
    trajectory = root / "artifacts/trajectories/v5" / f"{name}_validation.csv"
    if checkpoint.exists() or trajectory.exists():
        raise FileExistsError(f"V5 artifact exists: {name}")
    seed_everything(seed)
    model = build_v2_dqn(_environment(train_bundle, arm=arm, training=True), hp, seed=seed)
    before_parameters = parameter_sha256(model)
    before_scaler = scaler_sha256(train_bundle.observation_builder)
    callback = _callback()
    model.learn(total_timesteps=timesteps, callback=callback)
    if not callback.losses or not np.isfinite(callback.losses).all():
        raise RuntimeError("V5 training did not produce finite losses")
    if parameter_sha256(model) == before_parameters:
        raise RuntimeError("V5 model did not update")
    if scaler_sha256(train_bundle.observation_builder) != before_scaler:
        raise RuntimeError("V5 scaler mutated during training")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    env = _environment(val_bundle, arm=arm)
    loaded = DQN.load(checkpoint, env=env, device="cpu")
    rows, measures, error, responsiveness, unchanged = _eval_model(
        loaded, env, val_bundle.observation_builder, seed)
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(trajectory, index=False)
    return {
        "fold": fold, "seed": seed, "known_period_accessed": False,
        "action_weights": list(ACTION_WEIGHTS[arm]), "execution_rule": RULE,
        "technical_pass": bool(unchanged and abs(error) <= 1e-10),
        "state_responsive": bool(responsiveness["eligible"]),
        "responsiveness": responsiveness, "reward_nav_error": error,
        "evaluation_state_unchanged": unchanged,
        "checkpoint_path": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "trajectory_path": trajectory.relative_to(root).as_posix(),
        "trajectory_sha256": sha256_file(trajectory),
        **measures,
    }


def _baselines(bundle, seed):
    result = {}
    for name in BASELINES:
        env = _environment(bundle)
        if name == "cash":
            choose = lambda _obs, _i: 0
        elif name == "buy_and_hold":
            choose = lambda _obs, i: 2 if i == 0 else None
        else:
            choose = lambda _obs, _i: 1
        _rows, metrics, _error = rollout(env, choose, seed=seed)
        result[name] = metrics
    return result


def _reserve(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def run_arm(root: Path, data_root: Path, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("only frozen V5 arms are allowed")
    provenance = preflight(root, data_root)
    attempt = root / f"experiments/v5_attempts/{arm}.json"
    record_path = root / f"experiments/v5_records/{arm}.json"
    if attempt.exists() or record_path.exists():
        raise FileExistsError("V5 attempt/record already exists; inspect before retry")
    for folder, pattern in (("artifacts/checkpoints/v5", f"v5_{arm}_*.zip"),
                            ("artifacts/trajectories/v5", f"v5_{arm}_*.csv")):
        if list((root / folder).glob(pattern)):
            raise FileExistsError("orphan V5 artifact exists; inspect before retry")
    market, events, folds, hp = _inputs(root, data_root)
    _reserve(attempt, {"status": "reserved", "arm": arm, **provenance})
    results, baselines = [], {}
    status, error = "FAIL", None
    try:
        for fold in folds.folds:
            split = folds.split_contract_for(fold)
            train = prepare_asset_bundle(market, events, split, ticker="SPY", partition="train")
            val = prepare_asset_bundle(market, events, split, ticker="SPY", partition="validation")
            baselines[fold.fold_id] = _baselines(val, SEEDS[0])
            for seed in SEEDS:
                print(f"V5 {arm} {fold.fold_id} seed {seed}: training 200000 steps", flush=True)
                row = _train_one(root, arm, hp, fold.fold_id, seed, train, val,
                                 timesteps=200000)
                results.append(row)
                print(f"V5 {arm} {fold.fold_id} seed {seed}: "
                      f"technical_pass={row['technical_pass']}, "
                      f"Sharpe={row['sharpe_excess_cash']}", flush=True)
        status = "PASS" if all(row["technical_pass"] for row in results) else "FAIL"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record = {
            "status": status, "error": error, "arm": arm,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "folds": list(FOLDS), "seeds": list(SEEDS),
            "timesteps_per_fold_seed": 200000,
            "known_period_accessed": False,
            "preflight": provenance,
            "baseline_metrics": baselines,
            "per_fold_seed": results,
        }
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
    return record


def aggregate(root: Path) -> dict:
    """Read V5 records only. Require exact provenance and all paired runs."""
    contract(root)
    records = {}
    for arm in ARMS:
        record = json.loads((root / f"experiments/v5_records/{arm}.json").read_text(encoding="utf-8"))
        if record["arm"] != arm or record["status"] != "PASS" or record["known_period_accessed"] is not False:
            raise ValueError(f"V5 arm integrity failed: {arm}")
        if record["folds"] != list(FOLDS) or record["seeds"] != list(SEEDS) or record["timesteps_per_fold_seed"] != 200000:
            raise ValueError("V5 paired protocol mismatch")
        expected_pairs = {(f, s) for f in FOLDS for s in SEEDS}
        actual_pairs = [(r["fold"], r["seed"]) for r in record["per_fold_seed"]]
        if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
            raise ValueError("V5 missing or duplicate fold/seed")
        if not all(r["technical_pass"] and r["known_period_accessed"] is False for r in record["per_fold_seed"]):
            raise ValueError("V5 technical run failed")
        for row in record["per_fold_seed"]:
            if row["action_weights"] != list(ACTION_WEIGHTS[arm]) or row["execution_rule"] != RULE:
                raise ValueError("V5 action contract mismatch")
            for kind in ("checkpoint", "trajectory"):
                rel = Path(row[f"{kind}_path"])
                if rel.is_absolute() or ".." in rel.parts or sha256_file(root / rel) != row[f"{kind}_sha256"]:
                    raise ValueError(f"V5 {kind} hash mismatch")
        records[arm] = record
    a, b = (records[arm] for arm in ARMS)
    if a["preflight"] != b["preflight"] or a["baseline_metrics"] != b["baseline_metrics"]:
        raise ValueError("V5 provenance or baseline mismatch")
    if a["preflight"]["git_commit"] != _git_commit(root):
        raise ValueError("V5 records are not from current code commit")
    pairs = []
    control = {(r["fold"], r["seed"]): r for r in a["per_fold_seed"]}
    for row in b["per_fold_seed"]:
        parent = control[(row["fold"], row["seed"])]
        pairs.append({
            "fold": row["fold"], "seed": row["seed"],
            "parent_state_responsive": parent["state_responsive"],
            "floor_state_responsive": row["state_responsive"],
            **{f"paired_{key}_difference":
               (row[key] - parent[key] if row[key] is not None and parent[key] is not None else None)
               for key in ("sharpe_excess_cash", "annualized_mean_excess_return", "total_turnover",
                           "final_wealth", "max_drawdown", "average_exposure")},
        })
    summary = {
        "status": "PASS", "known_period_accessed": False,
        "provenance": a["preflight"],
        "arms": {arm: {
            "median_sharpe": float(np.median([r["sharpe_excess_cash"] for r in rec["per_fold_seed"]
                                              if r["sharpe_excess_cash"] is not None])),
            "median_annualized_excess_return": float(np.median([
                r["annualized_mean_excess_return"] for r in rec["per_fold_seed"]])),
            "median_turnover": float(np.median([r["total_turnover"] for r in rec["per_fold_seed"]])),
            "median_final_wealth": float(np.median([r["final_wealth"] for r in rec["per_fold_seed"]])),
            "median_max_drawdown": float(np.median([r["max_drawdown"] for r in rec["per_fold_seed"]])),
            "median_average_exposure": float(np.median([r["average_exposure"] for r in rec["per_fold_seed"]])),
            "state_responsive_count": sum(r["state_responsive"] for r in rec["per_fold_seed"]),
        } for arm, rec in records.items()},
        "baseline_metrics": a["baseline_metrics"], "paired_rows": pairs,
    }
    out = root / "reports/v5/PAIRED_VALIDATION.json"
    markdown = root / "reports/v5/PAIRED_VALIDATION.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or markdown.exists():
        raise FileExistsError("V5 aggregate exists; do not overwrite")
    text = [
        "# V5 pre-2022 paired validation — exposure floor",
        "",
        f"Git commit: `{a['preflight']['git_commit']}`; folds: `{list(FOLDS)}`; seeds: `{list(SEEDS)}`.",
        "Known-period metrics computed: **No**. All 15 fold/seed pairs retained.",
        "Both arms retrained under the same V4 clock and V5 commit. Earlier V4 runs are context only.",
        "",
        "| V5 arm | Action weights | Median Sharpe vs cash | Median annualized excess | Median wealth | Median max DD | Median exposure | Median turnover | Responsive / 15 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in summary["arms"].items():
        text.append(f"| {arm} | {list(ACTION_WEIGHTS[arm])} | {values['median_sharpe']:.3f} | "
                    f"{values['median_annualized_excess_return']:.2%} | "
                    f"{values['median_final_wealth']:.3f}× | {values['median_max_drawdown']:.2%} | "
                    f"{values['median_average_exposure']:.2%} | "
                    f"{values['median_turnover']:.3f} | {values['state_responsive_count']} / 15 |")
    text += ["", "## Same-clock deterministic baselines by fold", "",
             "| Fold | Policy | Sharpe vs cash | Annualized excess | Final wealth | Turnover |",
             "|---|---|---:|---:|---:|---:|"]
    for fold in FOLDS:
        for name in BASELINES:
            item = summary["baseline_metrics"][fold][name]
            sharpe = "NA" if item["sharpe_excess_cash"] is None else f"{item['sharpe_excess_cash']:.3f}"
            text.append(f"| {fold} | {name} | {sharpe} | "
                        f"{item['annualized_mean_excess_return']:.2%} | "
                        f"{item['final_wealth']:.3f}× | {item['total_turnover']:.3f} |")
    text += ["", "## Paired differences (floor minus parent)", "",
             "| Fold | Seed | Δ Sharpe | Δ annualized excess | Δ wealth | Δ max DD | Δ exposure | Δ turnover | Both responsive |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in pairs:
        delta = row["paired_sharpe_excess_cash_difference"]
        sharpe = "NA" if delta is None else f"{delta:+.3f}"
        text.append(f"| {row['fold']} | {row['seed']} | {sharpe} | "
                    f"{row['paired_annualized_mean_excess_return_difference']:+.2%} | "
                    f"{row['paired_final_wealth_difference']:+.3f} | "
                    f"{row['paired_max_drawdown_difference']:+.2%} | "
                    f"{row['paired_average_exposure_difference']:+.2%} | "
                    f"{row['paired_total_turnover_difference']:+.3f} | "
                    f"{row['parent_state_responsive'] and row['floor_state_responsive']} |")
    text += ["", "Per-run NAV, costs, exposure, drawdown, actions and hashes are in the JSON "
             "records and local trajectories. Technical PASS does not imply economic success.", ""]
    with out.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    with markdown.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(text))
    return summary
