"""Schema-validated experiment manifest reader/writer.

Provides the single source of truth for experiment automation. All tools
read from experiment.yaml via this module. Handles:
- Pydantic v2 schema validation with status-gated required fields
- Status transitions with state machine enforcement
- Atomic writes via Redis lock + write-then-rename (FUSE-safe)
- PR description generation from manifest state
- DB claim lifecycle with TTL-based Redis locking

Usage:
    from gigaevo.experiment.manifest import load_manifest, set_status, update_manifest

    manifest = load_manifest("hover/feedback_softfit")
    set_status("hover/feedback_softfit", "running")
    update_manifest("hover/feedback_softfit", lambda raw: raw.update(...))
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
import yaml

from gigaevo.experiment.lock import (
    _acquire_lock,
    _get_redis,
    _release_lock,
    _write_manifest_atomic,
)

PROJ = Path(__file__).parent.parent.parent

SUPPORTED_SCHEMA_VERSIONS = {1}
VALID_STATUSES = {"preregistered", "implemented", "running", "complete", "invalid"}

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

DB_CLAIM_TTL = 86400 * 7


# ---------------------------------------------------------------------------
# Pydantic Schema Models
# ---------------------------------------------------------------------------


class PlotCommand(BaseModel):
    """A CLI plot command to invoke from the watchdog plugin."""

    model_config = ConfigDict(extra="ignore")

    command: str
    args: dict[str, Any] = {}
    output_name: str = ""
    caption: str = ""


class AlertThresholds(BaseModel):
    """Configurable alert thresholds for watchdog."""

    model_config = ConfigDict(extra="ignore")

    invalidity_rate: float = 0.75
    stagnation_window: int = 10
    generation_gap_threshold: int = 5


class WatchdogSection(BaseModel):
    """Watchdog configuration section in experiment.yaml."""

    model_config = ConfigDict(extra="ignore")

    plugin: str | None = None
    plot_commands: list[PlotCommand] = []
    plot_metrics: list[str] = []
    sentinel_value: float | None = None
    alert_thresholds: AlertThresholds = AlertThresholds()
    poll_interval_s: int = 3600
    plot_retries: int = 3
    plot_retry_delay_s: int = 30
    rolling_comment_threshold_hours: int = 24
    checkpoint_milestones: list[float] = [0.1, 0.2, 0.5, 1.0]
    no_proxy_hosts: list[str] = []


class RunSpec(BaseModel):
    """One run within an experiment."""

    model_config = ConfigDict(extra="ignore")

    label: str
    db: int
    prefix: str
    pipeline: str
    problem_name: str
    condition: str
    chain_url: str | None = None
    mutation_url: str | None = None
    model_name: str
    pid: int | None = None
    log_path: str | None = None
    extra_overrides: list[str] | None = None
    role: Literal["constructor", "improver"] | None = None

    @field_validator("db")
    @classmethod
    def db_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"db must be >= 0, got {v}")
        return v


class ProblemSpec(BaseModel):
    """Problem configuration section."""

    model_config = ConfigDict(extra="ignore")

    has_test_set: bool = True
    fitness_type: str = "discrete"
    metric_name: str = ""
    test_set_path: str | None = None
    test_set_sha256: str | None = None
    max_val_test_gap: float | None = None


class LaunchInfo(BaseModel):
    """Launch state information."""

    model_config = ConfigDict(extra="ignore")

    time: str | None = None
    commit: str | None = None
    watchdog_pid: int | None = None
    confirmed_at: str | None = None
    attempt: int | None = None


class BaselineInfo(BaseModel):
    """Baseline reference for comparison."""

    model_config = ConfigDict(extra="ignore")

    reference: str | None = None
    mean: float | None = None
    metric: str | None = None


class SmokeTestInfo(BaseModel):
    """Smoke test state."""

    model_config = ConfigDict(extra="ignore")

    completed: bool = False
    db: int | None = None
    generations: int = 3
    log_path: str | None = None
    completed_at: str | None = None


class TreatmentVerificationInfo(BaseModel):
    """Treatment verification state."""

    model_config = ConfigDict(extra="ignore")

    completed: bool = False
    completed_at: str | None = None
    note: str = ""


class ExperimentSection(BaseModel):
    """The 'experiment' section of experiment.yaml."""

    model_config = ConfigDict(extra="ignore")

    name: str
    task: str
    status: str
    branch: str = ""
    max_generations: int = 25
    pr_number: int | None = None
    tracking_issue: int | None = None
    prereg_commit: str | None = None
    stopping_rule: str = ""

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{v}'. Valid statuses: {sorted(VALID_STATUSES)}"
            )
        return v

    @field_validator("max_generations")
    @classmethod
    def validate_max_generations(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_generations must be >= 0, got {v}")
        return v


class ExperimentManifest(BaseModel):
    """Pydantic-validated schema for experiment.yaml.

    Status-gated validation: fields that are required depend on the
    current experiment status (preregistered < implemented < running < complete).
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    experiment: ExperimentSection
    problem: ProblemSpec = ProblemSpec()
    runs: list[RunSpec] = []
    servers: list[str] = []
    config: dict[str, Any] = {}
    custom_env: dict[str, str] = {}
    checkpoints: list[dict[str, Any]] = []
    launch: LaunchInfo = LaunchInfo()
    baseline: BaselineInfo = BaselineInfo()
    smoke_test: SmokeTestInfo = SmokeTestInfo()
    tools: list[dict[str, str]] = []
    watchdog: WatchdogSection = WatchdogSection()
    treatment_verification: TreatmentVerificationInfo = TreatmentVerificationInfo()

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: int) -> int:
        if v not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version: {v}. "
                f"Supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        return v

    @model_validator(mode="after")
    def validate_status_gates(self) -> ExperimentManifest:
        """Validate required fields based on experiment status."""
        status = self.experiment.status
        errors: list[str] = []

        # implemented+ requires runs, servers, config, smoke_test.completed
        if status in ("implemented", "running", "complete"):
            if not self.runs:
                errors.append(
                    f"runs[] must be non-empty for status={status}. "
                    f"Add at least one run configuration."
                )
            if not self.servers:
                errors.append(
                    f"servers[] must be non-empty for status={status}. "
                    f"Add the server hostnames used by this experiment."
                )
            if not self.config:
                errors.append(
                    f"config must be non-empty for status={status}. "
                    f"Add the shared Hydra config overrides."
                )
            if not self.smoke_test.completed:
                errors.append(
                    f"smoke_test.completed must be true for status={status}. "
                    f"Run a smoke test first."
                )

        # running+ requires launch info and PIDs
        if status in ("running", "complete"):
            if not self.launch.time:
                errors.append(
                    f"launch.time is required for status={status}. "
                    f"Set to the ISO timestamp of the launch."
                )
            if not self.launch.commit:
                errors.append(
                    f"launch.commit is required for status={status}. "
                    f"Set to the git commit hash at launch."
                )
            for run in self.runs:
                if run.pid is None:
                    errors.append(
                        f"runs[{run.label}].pid is required for status={status}. "
                        f"Record the PID after launching."
                    )

        if errors:
            raise ValueError(
                f"Manifest validation failed (status={status}):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        return self

    @model_validator(mode="after")
    def validate_adversarial_roles(self) -> ExperimentManifest:
        """When watchdog.plugin='adversarial', every run must declare a role."""
        if self.watchdog.plugin == "adversarial":
            missing = [r.label for r in self.runs if r.role is None]
            if missing:
                raise ValueError(
                    f"watchdog.plugin='adversarial' requires every run to set "
                    f"role: 'constructor' or 'improver'. Missing role on: "
                    f"{missing}"
                )
        return self

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentManifest:
        """Validate a raw dict and return an ExperimentManifest."""
        return cls.model_validate(raw)

    @classmethod
    def from_yaml(cls, yaml_content: str) -> ExperimentManifest:
        """Parse YAML string and validate."""
        try:
            raw = yaml.safe_load(yaml_content)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML syntax: {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError(
                "YAML content must be a mapping (dict), not a scalar or list"
            )

        return cls.from_dict(raw)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> ExperimentManifest:
        """Load and validate from a YAML file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No experiment.yaml at {path}. "
                f"Create one from experiments/_template/experiment.yaml"
            )
        try:
            content = path.read_text()
        except OSError as exc:
            raise ValueError(f"Cannot read {path}: {exc}") from exc

        try:
            return cls.from_yaml(content)
        except ValueError as exc:
            raise ValueError(
                f"Validation failed for {path}:\n{exc}\nRecovery: git checkout {path}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Export to a dict suitable for YAML serialization."""
        return self.model_dump(mode="python", exclude_none=False)


def export_json_schema(output_path: str | Path) -> None:
    """Export the ExperimentManifest JSON Schema to a file."""
    schema = ExperimentManifest.model_json_schema()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def experiment_dir(experiment: str) -> Path:
    """Return the experiment directory path."""
    return PROJ / "experiments" / experiment


def manifest_path(experiment: str) -> Path:
    """Return the path to experiment.yaml."""
    return experiment_dir(experiment) / "experiment.yaml"


# ---------------------------------------------------------------------------
# Load and Validate
# ---------------------------------------------------------------------------


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
    return ExperimentManifest.from_dict(raw)


# ---------------------------------------------------------------------------
# Status Transitions
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
# DB Claims
# ---------------------------------------------------------------------------


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
# PR Description Generation
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
    badge = _STATUS_BADGES.get(m.experiment.status, m.experiment.status)

    # Running badge includes generation info
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


__all__ = [
    "VALID_STATUSES",
    "VALID_TRANSITIONS",
    "RECOVERY_TRANSITIONS",
    "DB_CLAIM_TTL",
    "PROJ",
    "RunSpec",
    "ProblemSpec",
    "LaunchInfo",
    "BaselineInfo",
    "SmokeTestInfo",
    "ExperimentSection",
    "ExperimentManifest",
    "WatchdogSection",
    "AlertThresholds",
    "PlotCommand",
    "export_json_schema",
    "experiment_dir",
    "manifest_path",
    "load_manifest",
    "set_status",
    "update_manifest",
    "claim_dbs",
    "refresh_db_claims",
    "release_db_claims",
    "find_active_experiments",
    "has_test_set",
    "generate_pr_description",
    "_validate",
    "_write_manifest_atomic",
]
