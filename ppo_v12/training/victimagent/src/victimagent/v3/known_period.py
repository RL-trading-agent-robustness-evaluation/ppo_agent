"""Frozen V3 known-period protocol helpers; importing this never reads market data."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import yaml


class ProtocolError(RuntimeError):
    """The frozen V3 known-period contract was not met."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_versions() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "gymnasium", "stable-baselines3", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def load_config(root: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((root / "configs/v3_known_period.yaml").read_text(encoding="utf-8"))
    if (cfg["version"], cfg["status"], cfg["selected_rule"]) != (
        1, "frozen_before_known_period_access", "v3_min_hold_five"
    ):
        raise ProtocolError("V3 known-period configuration changed")
    if (cfg["seeds"], cfg["timesteps_per_seed"], cfg["transaction_cost_bps"]) != (
        [0, 1, 2, 3, 4], 200000, 10
    ):
        raise ProtocolError("V3 budget, seeds, or costs changed")
    if (cfg["asset"], cfg["selected_experiment_id"], cfg["baselines"],
        cfg["known_period_role"], cfg["output_prefix"]) != (
        "SPY", "v3_confirmation_v3_min_hold_five",
        ["cash", "buy_and_hold", "constant_50", "ma_20_60"],
        "previously_seen_in_v1_not_untouched_test", "v3_known_period_v1"
    ):
        raise ProtocolError("V3 asset, selection, baselines, or artifact identity changed")
    if (str(cfg["train_start"]), str(cfg["train_end"]), str(cfg["known_start"]), str(cfg["known_end"])) != (
        "2010-01-04", "2021-12-31", "2022-01-01", "2025-12-31"
    ):
        raise ProtocolError("V3 known-period boundaries changed")
    if cfg["model_selection_on_known_period"] is not False or cfg["automatic_rerun_after_failure"] is not False:
        raise ProtocolError("selection or rerun on the known period is forbidden")
    if float(cfg["reward_nav_invariant_tolerance"]) != 1e-10:
        raise ProtocolError("reward–NAV tolerance changed")
    return cfg


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def provenance(root: Path, data_root: Path) -> dict[str, Any]:
    paths = {
        "v3_config": root / "configs/v3_trade_decision.yaml",
        "known_config": root / "configs/v3_known_period.yaml",
        "v2_hp_config": root / "configs/dqn_v2_search.yaml",
        "rolling_folds": root / "configs/v2_rolling_folds.yaml",
        "market": data_root / "data/processed/market_base_with_splits.csv",
        "distributions": data_root / "data/interim/distribution_events.csv",
        "registry": root / "experiments/v3_registry.jsonl",
        "selection": root / "reports/v3/selection_confirmation.json",
    }
    return {key: sha256(value) for key, value in paths.items()}


def validate_selection(root: Path, hashes: dict[str, str], cfg: dict[str, Any]) -> None:
    if (hashes["registry"], hashes["selection"]) != (
        cfg["source_registry_sha256"], cfg["source_selection_sha256"]
    ):
        raise ProtocolError("source registry or selection has changed")
    from victimagent.v3.run import _records, load_contract, select_v3
    records = _records(root)
    result = select_v3(load_contract(root), records, "confirmation")
    disk = json.loads((root / "reports/v3/selection_confirmation.json").read_text(encoding="utf-8"))
    if result["winner_experiment_id"] != cfg["selected_experiment_id"] or disk["winner_experiment_id"] != result["winner_experiment_id"]:
        raise ProtocolError("V3 confirmation winner changed")
    confirmed = {r["experiment_id"]: r for r in records if r["stage"] == "confirmation"}
    chosen = confirmed[cfg["selected_experiment_id"]]
    control = confirmed["v3_confirmation_v3_control_daily_target"]
    if (chosen["status"], control["status"], chosen["git_commit"], control["git_commit"]) != (
        "pass", "fail", cfg["source_training_commit"], cfg["source_training_commit"]
    ):
        raise ProtocolError("V3 eligibility or source training commit changed")
    if chosen["data_sha256"] != control["data_sha256"] or (
        chosen["data_sha256"]["processed_market"] != hashes["market"] or
        chosen["data_sha256"]["distribution_events"] != hashes["distributions"]
    ):
        raise ProtocolError("V3 data snapshot differs from the frozen confirmation")
    if (chosen["config_sha256"], chosen["split_sha256"],
        chosen["v2_parent_hyperparameter_config_sha256"]) != (
        hashes["v3_config"], hashes["rolling_folds"], hashes["v2_hp_config"]
    ):
        raise ProtocolError("V3 config, split, or parent HP differs from the frozen confirmation")


def assert_research_clean(root: Path) -> str:
    status = git(root, "status", "--porcelain", "--", "src", "scripts", "configs", "pyproject.toml")
    if status:
        raise ProtocolError(f"uncommitted research code or config: {status}")
    return git(root, "rev-parse", "HEAD")


def train_paths(root: Path, seed: int) -> tuple[Path, Path, Path]:
    prefix = f"v3_known_period_v1_seed{seed}"
    return (
        root / f"experiments/v3_known/{prefix}_attempt.json",
        root / f"artifacts/checkpoints/v3_known/{prefix}.zip",
        root / f"experiments/v3_known/{prefix}_train.json",
    )


def output_paths(root: Path) -> dict[str, Path]:
    return {
        "preflight": root / "reports/v3_known/preflight.json",
        "execution": root / "reports/v3_known/execution_record.json",
        "report": root / "reports/v3_known/report.json",
        "trajectories": root / "artifacts/trajectories/v3_known",
    }


def no_previous_execution(root: Path) -> None:
    paths = output_paths(root)
    if any(paths[k].exists() for k in ("execution", "report")) or (
        paths["trajectories"].exists() and any(paths["trajectories"].iterdir())
    ):
        raise ProtocolError("V3 known period already has execution evidence; do not rerun")


def record_training(root: Path, seed: int, commit: str, hashes: dict[str, str],
                    checkpoint_hash: str, checks: dict[str, bool]) -> dict[str, Any]:
    if not checks or not all(checks.values()):
        raise ProtocolError(f"training checks failed: {checks}")
    item = {"seed": seed, "commit": commit, "input_sha256": hashes,
            "checkpoint_sha256": checkpoint_hash, "timesteps": 200000,
            "checks": checks, "known_period_accessed": False, "created_at_utc": now(),
            "runtime_versions": runtime_versions()}
    path = train_paths(root, seed)[2]
    with path.open("x", encoding="utf-8") as stream:
        json.dump(item, stream, indent=2)
        stream.write("\n")
    return item


def verify_training(root: Path, cfg: dict[str, Any], commit: str,
                    hashes: dict[str, str]) -> list[dict[str, Any]]:
    records = []
    for seed in cfg["seeds"]:
        attempt, checkpoint, path = train_paths(root, seed)
        if not attempt.exists() or not checkpoint.exists() or not path.exists():
            raise ProtocolError(f"missing completed training seed {seed}")
        marker = json.loads(attempt.read_text(encoding="utf-8"))
        item = json.loads(path.read_text(encoding="utf-8"))
        if (marker.get("seed"), marker.get("commit"), marker.get("input_sha256"),
            marker.get("known_period_accessed")) != (seed, commit, hashes, False):
            raise ProtocolError(f"training attempt marker mismatch for seed {seed}")
        if (item.get("seed"), item.get("commit"), item.get("timesteps"),
            item.get("input_sha256"), item.get("checkpoint_sha256"),
            item.get("known_period_accessed"), item.get("runtime_versions")) != (
            seed, commit, 200000, hashes, sha256(checkpoint), False,
            runtime_versions()
        ) or not item.get("checks") or not all(item["checks"].values()):
            raise ProtocolError(f"training provenance mismatch for seed {seed}")
        records.append(item)
    return records
