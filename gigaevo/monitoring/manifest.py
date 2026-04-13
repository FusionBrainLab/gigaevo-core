"""Pydantic manifest operations with Redis locking and atomic writes.

Wraps the Pydantic ExperimentManifest schema with mutation operations:
load, status transitions, generic update, DB claims, discovery, PR description.

All functions return ExperimentManifest (Pydantic), NOT the legacy dataclass.

Usage:
    from gigaevo.monitoring.manifest import load_manifest, set_status, update_manifest

    manifest = load_manifest("hover/feedback_softfit")
    set_status("hover/feedback_softfit", "running")
    update_manifest("hover/feedback_softfit", lambda m: m.update({"launch": {"time": "..."}}))
"""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import time
from typing import Any

import redis
import yaml

from gigaevo.monitoring.manifest_schema import ExperimentManifest

PROJ = Path(__file__).parent.parent.parent

VALID_TRANSITIONS: dict[str, set[str]] = {
    "preregistered": {"implemented"},
    "implemented": {"running"},
    "running": {"complete", "invalid"},
    "complete": set(),
    "invalid": {"preregistered"},
}

RECOVERY_TRANSITIONS: dict[str, set[str]] = {
    "running": {"implemented"},
}

_STATUS_BADGES = {
    "preregistered": "🔵 Pre-registered",
    "implemented": "🔵 Pre-registered (implemented, not launched)",
    "running": "🟡 Running",
    "complete": "🟢 Complete",
    "invalid": "🔴 Invalid",
}

DB_CLAIM_TTL = 86400 * 7


def experiment_dir(experiment: str) -> Path:
    return PROJ / "experiments" / experiment


def manifest_path(experiment: str) -> Path:
    return experiment_dir(experiment) / "experiment.yaml"


def _get_redis() -> redis.Redis:
    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    return redis.Redis(host=host, port=port, db=0)


def _acquire_lock(r: redis.Redis, experiment: str, timeout: float = 5.0) -> str:
    lock_key = f"experiments:{experiment}:yaml_lock"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if r.set(lock_key, str(os.getpid()), nx=True, ex=30):
            return lock_key
        time.sleep(0.25)
    raise RuntimeError(
        f"Could not acquire lock {lock_key} after {timeout}s. "
        f"Current holder: {r.get(lock_key)}"
    )


def _release_lock(r: redis.Redis, lock_key: str) -> None:
    r.delete(lock_key)


def _write_manifest_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


def load_manifest(experiment: str) -> ExperimentManifest:
    path = manifest_path(experiment)
    if not path.exists():
        raise FileNotFoundError(
            f"No experiment.yaml at {path}. "
            f"Create one from experiments/_template/experiment.yaml"
        )
    return ExperimentManifest.from_yaml_file(path)


def set_status(
    experiment: str,
    new_status: str,
    *,
    allow_recovery: bool = False,
) -> ExperimentManifest:
    r = _get_redis()
    lock_key = _acquire_lock(r, experiment)
    try:
        path = manifest_path(experiment)
        with open(path) as f:
            raw = yaml.safe_load(f)

        current = raw.get("experiment", {}).get("status", "preregistered")

        allowed = VALID_TRANSITIONS.get(current, set())
        if allow_recovery:
            allowed = allowed | RECOVERY_TRANSITIONS.get(current, set())

        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition: {current} -> {new_status}. Allowed: {allowed}"
            )

        raw.setdefault("experiment", {})["status"] = new_status
        manifest = ExperimentManifest.from_dict(raw)
        _write_manifest_atomic(path, raw)
        return manifest
    finally:
        _release_lock(r, lock_key)


def update_manifest(
    experiment: str,
    updater: Callable[[dict[str, Any]], None],
) -> ExperimentManifest:
    r = _get_redis()
    lock_key = _acquire_lock(r, experiment)
    try:
        path = manifest_path(experiment)
        with open(path) as f:
            raw = yaml.safe_load(f)

        updater(raw)

        manifest = ExperimentManifest.from_dict(raw)
        _write_manifest_atomic(path, raw)
        return manifest
    finally:
        _release_lock(r, lock_key)


def claim_dbs(experiment: str, dbs: list[int]) -> list[tuple[int, str]]:
    r = _get_redis()
    failed: list[tuple[int, str]] = []
    for db in dbs:
        key = f"experiments:db_claim:{db}"
        if not r.set(key, experiment, nx=True, ex=DB_CLAIM_TTL):
            owner = (r.get(key) or b"unknown").decode()
            if owner != experiment:
                failed.append((db, owner))
    return failed


def refresh_db_claims(experiment: str, dbs: list[int]) -> None:
    r = _get_redis()
    for db in dbs:
        key = f"experiments:db_claim:{db}"
        r.set(key, experiment, xx=True, ex=DB_CLAIM_TTL)


def release_db_claims(dbs: list[int]) -> None:
    r = _get_redis()
    for db in dbs:
        r.delete(f"experiments:db_claim:{db}")


def find_active_experiments() -> list[ExperimentManifest]:
    active: list[ExperimentManifest] = []
    experiments_dir = PROJ / "experiments"
    for yaml_path in experiments_dir.glob("*/*/experiment.yaml"):
        if "_template" in str(yaml_path):
            continue
        rel = yaml_path.parent.relative_to(experiments_dir)
        experiment = str(rel)
        try:
            m = load_manifest(experiment)
            if m.experiment.status in ("implemented", "running"):
                active.append(m)
        except (ValueError, yaml.YAMLError, FileNotFoundError):
            continue
    return active


def generate_pr_description(experiment: str) -> str:
    m = load_manifest(experiment)
    badge = _STATUS_BADGES.get(m.experiment.status, m.experiment.status)

    if m.experiment.status == "running" and m.checkpoints:
        last_cp = m.checkpoints[-1]
        gen = last_cp.get("gen", "?")
        badge = f"🟡 Running (gen {gen}/{m.experiment.max_generations})"
    elif m.experiment.status == "running":
        badge = f"🟡 Running (gen 0/{m.experiment.max_generations})"

    lines = [
        f"# exp: {m.experiment.name}",
        "",
        f"**Status**: {badge}",
        f"**Branch**: `{m.experiment.branch}`",
        f"**Tracking issue**: #{m.experiment.tracking_issue}"
        if m.experiment.tracking_issue
        else "",
        "",
        "## Design",
        "",
        f"See `experiments/{m.experiment.name}/01_design.md` for full design.",
        "",
        "## Runs",
        "",
        "| Label | DB | Condition | Pipeline | PID |",
        "|-------|----|-----------|----------|-----|",
    ]

    for run in m.runs:
        pid_str = str(run.pid) if run.pid else "-"
        lines.append(
            f"| {run.label} | {run.db} | {run.condition} | {run.pipeline} | {pid_str} |"
        )

    lines.extend(["", "## Checkpoints", ""])

    if m.checkpoints:
        lines.append("| Gen | Time | Notes |")
        lines.append("|-----|------|-------|")
        for cp in m.checkpoints:
            gen = cp.get("gen", "?")
            ts = cp.get("timestamp", "")
            notes = cp.get("notes", "")
            lines.append(f"| {gen} | {ts} | {notes} |")
    else:
        lines.append("_No checkpoints yet._")

    if m.baseline.reference:
        lines.extend(
            [
                "",
                "## Baseline",
                "",
                f"Reference: `{m.baseline.reference}` "
                f"(mean={m.baseline.mean}, metric={m.baseline.metric})",
            ]
        )

    lines.extend(["", "## Archives", "", "_(pending)_", ""])

    return "\n".join(line for line in lines if line is not None) + "\n"
