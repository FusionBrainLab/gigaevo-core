"""Schema-validated experiment manifest reader/writer.

Provides the single source of truth for experiment automation. All tools
read from experiment.yaml via this module. Handles:
- Schema validation with status-gated required fields
- Status transitions with state machine enforcement
- Atomic writes via Redis lock + write-then-rename (FUSE-safe)
- PR description generation from manifest state

Usage:
    from gigaevo.experiment.manifest import load_manifest, set_status, update_manifest

    manifest = load_manifest("hover/feedback_softfit")
    set_status("hover/feedback_softfit", "running")
    update_manifest("hover/feedback_softfit", lambda m: m.update({"launch": {"time": "..."}}))
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any

import redis
import yaml

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJ = Path(
    __file__
).parent.parent.parent  # gigaevo/experiment/manifest.py -> repo root


# ---------------------------------------------------------------------------
# Status state machine
# ---------------------------------------------------------------------------
VALID_STATUSES = {"preregistered", "implemented", "running", "complete", "invalid"}

VALID_TRANSITIONS: dict[str, set[str]] = {
    "preregistered": {"implemented"},
    "implemented": {"running"},
    "running": {"complete", "invalid"},
    "complete": set(),
    "invalid": {"preregistered"},  # allow retry after fixing
}

# Recovery transitions allowed only by `gigaevo manifest reset-status`
RECOVERY_TRANSITIONS: dict[str, set[str]] = {
    "running": {"implemented"},
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class RunSpec:
    label: str
    db: int
    prefix: str
    pipeline: str
    problem_name: str
    condition: str
    chain_url: str
    mutation_url: str
    model_name: str
    pid: int | None = None
    log_path: str | None = None
    extra_overrides: list[str] | None = None
    run_env: dict[str, str] | None = (
        None  # per-run env vars prepended to launch command
    )


@dataclass
class ProblemSpec:
    has_test_set: bool
    fitness_type: str
    metric_name: str
    test_set_path: str | None = None
    test_set_sha256: str | None = None
    max_val_test_gap: float | None = None


@dataclass
class LaunchInfo:
    time: str | None = None
    commit: str | None = None
    watchdog_pid: int | None = None
    confirmed_at: str | None = None
    attempt: int | None = None


@dataclass
class BaselineInfo:
    reference: str | None = None
    mean: float | None = None
    metric: str | None = None


@dataclass
class SmokeTestInfo:
    completed: bool = False
    db: int | None = None
    generations: int = 3
    log_path: str | None = None
    completed_at: str | None = None


@dataclass
class ExperimentManifest:
    schema_version: int
    name: str
    task: str
    status: str
    max_generations: int
    branch: str
    problem: ProblemSpec
    runs: list[RunSpec] = field(default_factory=list)
    servers: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    custom_env: dict[str, str] = field(default_factory=dict)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    launch: LaunchInfo = field(default_factory=LaunchInfo)
    baseline: BaselineInfo = field(default_factory=BaselineInfo)
    smoke_test: SmokeTestInfo = field(default_factory=SmokeTestInfo)
    tools: list[dict[str, str]] = field(default_factory=list)
    pr_number: int | None = None
    tracking_issue: int | None = None
    prereg_commit: str | None = None
    watchdog_plugin: str | None = None

    # ---- Raw dict for round-trip fidelity ----
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_run(d: dict[str, Any]) -> RunSpec:
    return RunSpec(
        label=d["label"],
        db=int(d["db"]),
        prefix=d["prefix"],
        pipeline=d["pipeline"],
        problem_name=d["problem_name"],
        condition=d["condition"],
        chain_url=d["chain_url"],
        mutation_url=d["mutation_url"],
        model_name=d["model_name"],
        pid=d.get("pid"),
        log_path=d.get("log_path"),
        extra_overrides=d.get("extra_overrides"),
        run_env=d.get("run_env"),
    )


def _parse_problem(d: dict[str, Any]) -> ProblemSpec:
    return ProblemSpec(
        has_test_set=bool(d.get("has_test_set", True)),
        fitness_type=d.get("fitness_type", "discrete"),
        metric_name=d.get("metric_name", ""),
        test_set_path=d.get("test_set_path"),
        test_set_sha256=d.get("test_set_sha256"),
        max_val_test_gap=d.get("max_val_test_gap"),
    )


def _parse_launch(d: dict[str, Any] | None) -> LaunchInfo:
    if not d:
        return LaunchInfo()
    return LaunchInfo(
        time=d.get("time"),
        commit=d.get("commit"),
        watchdog_pid=d.get("watchdog_pid"),
        confirmed_at=d.get("confirmed_at"),
    )


def _parse_baseline(d: dict[str, Any] | None) -> BaselineInfo:
    if not d:
        return BaselineInfo()
    return BaselineInfo(
        reference=d.get("reference"),
        mean=d.get("mean"),
        metric=d.get("metric"),
    )


def _parse_smoke_test(d: dict[str, Any] | None) -> SmokeTestInfo:
    if not d:
        return SmokeTestInfo()
    return SmokeTestInfo(
        completed=bool(d.get("completed", False)),
        db=d.get("db"),
        generations=d.get("generations", 3),
        log_path=d.get("log_path"),
        completed_at=d.get("completed_at"),
    )


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def experiment_dir(experiment: str) -> Path:
    """Return the experiment directory path."""
    return PROJ / "experiments" / experiment


def manifest_path(experiment: str) -> Path:
    """Return the path to experiment.yaml."""
    return experiment_dir(experiment) / "experiment.yaml"


def load_manifest(experiment: str) -> ExperimentManifest:
    """Load and validate experiment.yaml.

    Raises FileNotFoundError if the file doesn't exist.
    Raises ValueError on schema validation failure.
    Raises yaml.YAMLError on parse failure (with path hint).
    """
    path = manifest_path(experiment)
    if not path.exists():
        raise FileNotFoundError(
            f"No experiment.yaml at {path}. "
            f"Create one from experiments/_template/experiment.yaml"
        )

    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Failed to parse {path}: {e}\n"
            f"Recovery: git checkout {path.relative_to(PROJ)}"
        ) from e

    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a YAML mapping")

    return _validate(raw, experiment)


def _validate(raw: dict[str, Any], experiment: str) -> ExperimentManifest:
    """Validate raw YAML dict and return ExperimentManifest."""
    # Schema version
    sv = raw.get("schema_version", 1)
    if sv != 1:
        raise ValueError(f"Unsupported schema_version: {sv} (expected 1)")

    # Experiment section
    exp = raw.get("experiment", {})
    name = exp.get("name", experiment)
    task = exp.get("task", "")
    status = exp.get("status", "preregistered")
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Valid: {VALID_STATUSES}")

    branch = exp.get("branch", "")
    max_gen = exp.get("max_generations", 25)

    # Problem section
    problem = _parse_problem(raw.get("problem", {}))

    # Runs
    runs_raw = raw.get("runs") or []
    runs = [_parse_run(r) for r in runs_raw]

    # Other sections
    servers = raw.get("servers") or []
    config = raw.get("config") or {}
    custom_env = raw.get("custom_env") or {}
    checkpoints = raw.get("checkpoints") or []
    launch = _parse_launch(raw.get("launch"))
    baseline = _parse_baseline(raw.get("baseline"))
    smoke_test = _parse_smoke_test(raw.get("smoke_test"))
    tools = raw.get("tools") or []

    manifest = ExperimentManifest(
        schema_version=sv,
        name=name,
        task=task,
        status=status,
        max_generations=max_gen,
        branch=branch,
        problem=problem,
        runs=runs,
        servers=servers,
        config=config,
        custom_env=custom_env,
        checkpoints=checkpoints,
        launch=launch,
        baseline=baseline,
        smoke_test=smoke_test,
        tools=tools,
        pr_number=exp.get("pr_number"),
        tracking_issue=exp.get("tracking_issue"),
        prereg_commit=exp.get("prereg_commit"),
        watchdog_plugin=raw.get("watchdog_plugin")
        or raw.get("watchdog", {}).get("plugin"),
        _raw=raw,
    )

    # Status-gated required fields
    _validate_for_status(manifest)
    return manifest


def _validate_for_status(m: ExperimentManifest) -> None:
    """Check required fields based on current status."""
    errors: list[str] = []

    # All statuses need basic experiment info
    if not m.name:
        errors.append("experiment.name is required")
    if not m.task:
        errors.append("experiment.task is required")

    # implemented+ needs runs, servers, config
    if m.status in ("implemented", "running", "complete"):
        if not m.runs:
            errors.append(f"runs[] must be non-empty for status={m.status}")
        if not m.servers:
            errors.append(f"servers[] must be non-empty for status={m.status}")
        if not m.config:
            errors.append(f"config must be non-empty for status={m.status}")
        if not m.smoke_test.completed:
            errors.append(f"smoke_test.completed must be true for status={m.status}")

    # running+ needs launch info
    if m.status in ("running", "complete"):
        if not m.launch.time:
            errors.append(f"launch.time is required for status={m.status}")
        if not m.launch.commit:
            errors.append(f"launch.commit is required for status={m.status}")
        for run in m.runs:
            if run.pid is None:
                errors.append(
                    f"runs[{run.label}].pid is required for status={m.status}"
                )

    if errors:
        raise ValueError(
            f"Validation errors for {m.name} (status={m.status}):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# Redis locking (FUSE-safe — replaces fcntl.flock)
# ---------------------------------------------------------------------------


def _get_redis() -> redis.Redis:
    return redis.Redis(host="localhost", port=6379, db=0)


def _acquire_lock(r: redis.Redis, experiment: str, timeout: float = 5.0) -> str:
    """Acquire Redis-based lock. Returns lock key. Raises on timeout."""
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
    """Write YAML atomically via tmp + rename."""
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Mutation API
# ---------------------------------------------------------------------------


def set_status(
    experiment: str,
    new_status: str,
    *,
    allow_recovery: bool = False,
) -> ExperimentManifest:
    """Transition experiment to new_status. Validates transition.

    Args:
        experiment: e.g. "hover/feedback_softfit"
        new_status: target status
        allow_recovery: if True, also allow RECOVERY_TRANSITIONS

    Returns:
        Updated manifest.

    Raises:
        ValueError on invalid transition or validation failure.
    """
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

        # Validate the new state (will raise if required fields missing)
        manifest = _validate(raw, experiment)

        _write_manifest_atomic(path, raw)
        return manifest
    finally:
        _release_lock(r, lock_key)


def update_manifest(
    experiment: str,
    updater: Callable[[dict[str, Any]], None],
) -> ExperimentManifest:
    """Read-modify-write experiment.yaml under lock.

    The updater function receives the raw dict and modifies it in-place.
    Validation runs after the update.

    Usage:
        def set_pids(raw):
            for run in raw["runs"]:
                if run["label"] == "F1":
                    run["pid"] = 12345

        update_manifest("hover/feedback_softfit", set_pids)
    """
    r = _get_redis()
    lock_key = _acquire_lock(r, experiment)
    try:
        path = manifest_path(experiment)
        with open(path) as f:
            raw = yaml.safe_load(f)

        updater(raw)

        manifest = _validate(raw, experiment)
        _write_manifest_atomic(path, raw)
        return manifest
    finally:
        _release_lock(r, lock_key)


# ---------------------------------------------------------------------------
# DB claims (Redis SET NX)
# ---------------------------------------------------------------------------

DB_CLAIM_TTL = 86400 * 7  # 7 days


def claim_dbs(experiment: str, dbs: list[int]) -> list[tuple[int, str]]:
    """Atomically claim Redis DBs. Returns list of (db, owner) that failed."""
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
    """Refresh TTL on already-claimed DBs (called by watchdog each cycle)."""
    r = _get_redis()
    for db in dbs:
        key = f"experiments:db_claim:{db}"
        r.set(key, experiment, xx=True, ex=DB_CLAIM_TTL)


def release_db_claims(dbs: list[int]) -> None:
    """Release DB claims (called by reset_status and closeout)."""
    r = _get_redis()
    for db in dbs:
        r.delete(f"experiments:db_claim:{db}")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_active_experiments() -> list[ExperimentManifest]:
    """Scan for experiments with status in (implemented, running).

    Includes 'implemented' to prevent TOCTOU race with DB claims.
    """
    active: list[ExperimentManifest] = []
    experiments_dir = PROJ / "experiments"
    for yaml_path in experiments_dir.glob("*/*/experiment.yaml"):
        # Skip template
        if "_template" in str(yaml_path):
            continue
        rel = yaml_path.parent.relative_to(experiments_dir)
        experiment = str(rel)
        try:
            m = load_manifest(experiment)
            if m.status in ("implemented", "running"):
                active.append(m)
        except (ValueError, yaml.YAMLError, FileNotFoundError):
            continue
    return active


def has_test_set(experiment: str) -> bool:
    """Quick check: does this experiment have a test set?"""
    m = load_manifest(experiment)
    return m.problem.has_test_set


# ---------------------------------------------------------------------------
# PR description generation (P1)
# ---------------------------------------------------------------------------

_STATUS_BADGES = {
    "preregistered": "🔵 Pre-registered",
    "implemented": "🔵 Pre-registered (implemented, not launched)",
    "running": "🟡 Running",
    "complete": "🟢 Complete",
    "invalid": "🔴 Invalid",
}


def generate_pr_description(experiment: str) -> str:
    """Generate PR_DESCRIPTION.md content from experiment.yaml."""
    m = load_manifest(experiment)
    badge = _STATUS_BADGES.get(m.status, m.status)

    # Running badge includes generation info
    if m.status == "running" and m.checkpoints:
        last_cp = m.checkpoints[-1]
        gen = last_cp.get("gen", "?")
        badge = f"🟡 Running (gen {gen}/{m.max_generations})"
    elif m.status == "running":
        badge = f"🟡 Running (gen 0/{m.max_generations})"

    lines = [
        f"# exp: {m.name}",
        "",
        f"**Status**: {badge}",
        f"**Branch**: `{m.branch}`",
        f"**Tracking issue**: #{m.tracking_issue}" if m.tracking_issue else "",
        "",
        "## Design",
        "",
        f"See `experiments/{m.name}/01_design.md` for full design.",
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

    # Filter empty lines from conditional sections
    return "\n".join(line for line in lines if line is not None) + "\n"
