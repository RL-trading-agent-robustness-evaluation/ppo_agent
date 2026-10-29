"""Frozen Gate C2 contract and one-shot execution-record helpers.

This module contains no market-data loading.  In particular, importing it or
running its unit tests cannot inspect the locked test partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class GateC2ProtocolError(RuntimeError):
    """Raised when the frozen one-shot protocol is violated."""


@dataclass(frozen=True)
class GateC2Config:
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["selected_checkpoint"]["seed"])

    @property
    def checkpoint_path(self) -> str:
        return str(self.raw["selected_checkpoint"]["path"])

    @property
    def checkpoint_sha256(self) -> str:
        return str(self.raw["selected_checkpoint"]["sha256"])

    @property
    def confirmation_phrase(self) -> str:
        return str(self.raw["execution"]["confirmation_phrase"])

    def validate(self) -> None:
        r = self.raw
        if (int(r["version"]), r["status"], r["gate"]) != (1, "frozen", "C2"):
            raise ValueError("expected frozen Gate C2 V1 config")
        selected = r["selected_checkpoint"]
        if int(selected["seed"]) != 4:
            raise ValueError("Gate C2 V1 is frozen to selected seed 4")
        if selected["selection_metric"] != "validation_sharpe_after_costs":
            raise ValueError("checkpoint selection metric changed")
        if len(str(selected["sha256"])) != 64:
            raise ValueError("checkpoint SHA-256 is invalid")
        evaluation = r["evaluation"]
        if (evaluation["asset"], evaluation["partition"]) != ("SPY", "test"):
            raise ValueError("Gate C2 V1 evaluates SPY test only")
        if not bool(evaluation["deterministic"]):
            raise ValueError("Gate C2 must be deterministic")
        if float(evaluation["transaction_cost_bps"]) != 10.0:
            raise ValueError("Gate C2 cost must remain 10 bps")
        if tuple(evaluation["baselines"]) != (
            "cash",
            "buy_and_hold",
            "constant_50",
            "ma_20_60",
        ):
            raise ValueError("Gate C2 baseline set changed")
        if evaluation["economic_pass_threshold"] is not None:
            raise ValueError("Gate C2 has no economic-performance pass threshold")
        execution = r["execution"]
        if execution["confirmation_phrase"] != "I_UNDERSTAND_TEST_IS_ONE_SHOT":
            raise ValueError("Gate C2 confirmation phrase changed")
        if not bool(execution["refuse_if_execution_record_exists"]):
            raise ValueError("Gate C2 must refuse a second execution")
        if bool(execution["automatic_rerun_after_failure"]):
            raise ValueError("Gate C2 cannot automatically rerun after failure")


def load_gate_c2_config(path: str | Path) -> GateC2Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = GateC2Config(yaml.safe_load(handle))
    config.validate()
    return config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def claim_one_shot_execution(
    path: str | Path,
    *,
    git_commit: str,
    preflight_sha256: str,
) -> dict[str, Any]:
    """Atomically claim the sole formal execution before test data are loaded."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "STARTED",
        "started_at_utc": utc_now(),
        "git_commit": git_commit,
        "preflight_sha256": preflight_sha256,
        "test_partition_accessed": False,
        "automatic_rerun_permitted": False,
    }
    try:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise GateC2ProtocolError(
            f"Gate C2 execution record already exists: {target}. "
            "A second formal execution is forbidden."
        ) from exc
    return payload


def finalize_execution(
    path: str | Path,
    started: dict[str, Any],
    *,
    status: str,
    test_partition_accessed: bool,
    report_sha256: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if status not in {"PASS", "FAIL", "ERROR"}:
        raise ValueError("invalid final Gate C2 status")
    payload = {
        **started,
        "status": status,
        "completed_at_utc": utc_now(),
        "test_partition_accessed": bool(test_partition_accessed),
        "report_sha256": report_sha256,
        "error": error,
        "automatic_rerun_permitted": False,
    }
    write_json_atomic(path, payload)
    return payload
