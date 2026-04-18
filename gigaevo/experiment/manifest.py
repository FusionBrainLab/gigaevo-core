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
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml

from gigaevo.experiment.lock import (
    _acquire_lock,
    _get_redis,
    _release_lock,
    _write_manifest_atomic,
)


class RunRole(StrEnum):
    CONSTRUCTOR = "constructor"
    IMPROVER = "improver"


PROJ = Path(__file__).parent.parent.parent

SUPPORTED_SCHEMA_VERSIONS = {2}
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


class RunSpec(BaseModel):
    """One run within an experiment."""

    model_config = ConfigDict(extra="ignore")

    label: str
    db: int
    prefix: str
    pipeline: str
    problem_name: str
    condition: str
    pid: int | None = None
    log_path: str | None = None
    extra_overrides: list[str] | None = None
    role: RunRole | None = None
    pinned: dict[str, Any] = Field(default_factory=dict)
    # Per-run delta merged on top of contract.config.pinned. Lets one arm
    # assert a different resolved value without touching the others.

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
    """Launch state information recorded when an experiment transitions to
    ``running``. Live process/cron handles (``watchdog_pid``,
    ``anomaly_detector_cron_id``, ``checkpoint_cron_id``) live under
    ``control_plane`` and are intentionally absent here.
    """

    model_config = ConfigDict(extra="ignore")

    time: str | None = None
    commit: str | None = None
    confirmed_at: str | None = None
    attempt: int | None = None
    config_fingerprint: dict[str, str] = Field(default_factory=dict)
    # {relative_config_path: sha256 hex}. Populated by launch.py after the
    # dry-run + pin check pass. A relaunch refuses if any hash differs.


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
    note: str | None = None


# ---------------------------------------------------------------------------
# New typed sub-models for schema v2 (added additively in step 3).
#
# These are not yet wired into ExperimentManifest — step 4 will introduce the
# ContractSection / LifecycleState / TelemetryLog / ControlPlane groups that
# use them. For now they exist as first-class, round-trippable types so
# downstream code (migration script, checks, checkpoint skill) can start
# consuming them in subsequent steps.
# ---------------------------------------------------------------------------


class RunMetric(BaseModel):
    """One row inside ``CheckpointEntry.run_metrics``.

    Metric field names vary by problem (``best_fitness``, ``best_actual_fitness``,
    ``best_retrieval_coverage``, ...). We keep ``extra="allow"`` so the field
    round-trips unchanged regardless of problem type.
    """

    model_config = ConfigDict(extra="allow")

    label: str
    gen: int
    recent_invalidity: float | None = None


class CheckpointEntry(BaseModel):
    """One entry in the manifest's ``checkpoints`` list."""

    model_config = ConfigDict(extra="ignore")

    gen: int
    timestamp: str
    run_metrics: list[RunMetric] = []
    notes: str = ""
    metric_name: str | None = None


class MidRunTestEvalInfo(BaseModel):
    """State for the mid-run test set evaluation gate.

    Shape varies across v1 yamls: some use a ``results`` dict per run, others
    write free-form ``notes``. ``extra="allow"`` keeps both shapes safe.
    """

    model_config = ConfigDict(extra="allow")

    completed: bool = False
    completed_at: str | None = None
    notes: str = ""
    results: dict[str, Any] | None = None


class CheckpointAnalysisEntry(BaseModel):
    """Per-stage analyst entry (e.g. ``mid_run``)."""

    model_config = ConfigDict(extra="ignore")

    completed: bool = False
    completed_at: str | None = None
    summary: str = ""
    notes: str = ""


class CheckpointAnalysisInfo(BaseModel):
    """Checkpoint analyst state, keyed by stage.

    ``extra="allow"`` so future stages (``final``, ``post_freeze``, ...) round-trip
    without schema changes.
    """

    model_config = ConfigDict(extra="allow")

    mid_run: CheckpointAnalysisEntry = CheckpointAnalysisEntry()


class CheckResult(BaseModel):
    """Result of one treatment verification check."""

    model_config = ConfigDict(extra="ignore")

    name: str
    passed: bool
    detail: str = ""


class TreatmentChecksInfo(BaseModel):
    """Treatment-check verification state."""

    model_config = ConfigDict(extra="ignore")

    completed: bool = False
    completed_at: str | None = None
    results: list[CheckResult] = []


class ToolRef(BaseModel):
    """Reference to an auxiliary tool/script used by the experiment."""

    model_config = ConfigDict(extra="ignore")

    name: str
    path: str = ""
    purpose: str = ""


class ConfigSpec(BaseModel):
    """Shared Hydra config overrides.

    Standard keys are typed; problem-specific extras live in ``extra`` so the
    model doesn't have to know every Hydra override that experiments use.
    """

    model_config = ConfigDict(extra="ignore")

    problem_name: str | None = None
    pipeline: str | None = None
    prompt_fetcher: str | None = None
    evolution: str | None = None
    llm_model: str | None = None
    n_workers: int | None = None
    max_generations: int | None = None
    extra: dict[str, Any] = {}

    task_group: str | None = None
    # Name of a Hydra experiment group at config/experiment/<name>.yaml.
    # launch_generator emits ``experiment=<task_group>`` as the first CLI
    # override; Hydra composes the group file (which inherits ``base``) and
    # scalar overrides on the CLI win via Hydra's normal resolution rules.
    # ``None`` = falls through to config.yaml's default (``experiment: base``).

    pinned: dict[str, Any] = Field(default_factory=dict)
    # Flat dict: dotted Hydra path -> required resolved value.
    # Assertion contract — drift between pinned and resolved -> CRITICAL.
    # Missing key = no assertion on that path.

    @field_validator("pinned")
    @classmethod
    def validate_pinned_keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Reject keys containing shell metachars / CLI-override syntax.

        Keys must be bare dotted Hydra paths. The dry-run shells out to
        ``python run.py`` so any shell-active character is unsafe.
        CLI-override syntax (``=``, leading ``+``) would also break the
        launch_generator's arg emission.
        """
        _FORBIDDEN_CHARS = set(";|&`$><\n\r\t \"'\\")
        _FORBIDDEN_SUBSTRS = ("$(", "${")
        for key in v:
            if not isinstance(key, str) or not key:
                raise ValueError(f"pinned key must be a non-empty string, got {key!r}")
            if "\n" in key or "\r" in key:
                raise ValueError(
                    f"pinned key {key!r} contains newline; expected bare dotted Hydra path"
                )
            if "=" in key:
                raise ValueError(
                    f"pinned key {key!r} contains '='; pinned keys are paths, "
                    f"not override expressions (use value side for the '=')"
                )
            if key.startswith(("+", "~", "++")):
                raise ValueError(
                    f"pinned key {key!r} starts with Hydra override prefix; "
                    f"pinned keys are bare dotted paths"
                )
            if any(c in _FORBIDDEN_CHARS for c in key):
                raise ValueError(
                    f"pinned key {key!r} contains shell metachars; "
                    f"expected bare dotted Hydra path"
                )
            if any(s in key for s in _FORBIDDEN_SUBSTRS):
                raise ValueError(
                    f"pinned key {key!r} contains shell expansion; "
                    f"expected bare dotted Hydra path"
                )
        return v


class PrChannelConfig(BaseModel):
    """PR notification channel configuration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    comment_mode: Literal["new", "rolling"] = "rolling"


class TelegramChannelConfig(BaseModel):
    """Telegram notification channel configuration.

    Token and chat-id are always read from the environment — the manifest
    only records which env-var names to consult, never the secrets themselves.
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    chat_id_env: str = "TELEGRAM_CHAT_ID"
    token_env: str = "TELEGRAM_BOT_TOKEN"


class NotificationsSection(BaseModel):
    """Notification channel configuration (v2)."""

    model_config = ConfigDict(extra="ignore")

    pr: PrChannelConfig = PrChannelConfig()
    telegram: TelegramChannelConfig = TelegramChannelConfig()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Top-level sub-model groups.
#
#   contract      — pre-registered, researcher-authored, frozen at preregistered
#   lifecycle     — operational state (status, launch, smoke test, treatment verify)
#   telemetry     — append-only runtime records (checkpoints, analyses, checks)
#   control_plane — live processes + dashboards (watchdog, notifications, crons)
# ---------------------------------------------------------------------------


class ExperimentIdentity(BaseModel):
    """Pre-registered identity of an experiment (immutable after preregistration)."""

    model_config = ConfigDict(extra="ignore")

    name: str
    task: str
    branch: str = ""
    prereg_commit: str | None = None
    pr_number: int | None = None
    tracking_issue: int | None = None


class ContractSection(BaseModel):
    """The scientific contract — what was pre-registered and must not drift."""

    model_config = ConfigDict(extra="ignore")

    identity: ExperimentIdentity
    problem: ProblemSpec = ProblemSpec()
    config: ConfigSpec = ConfigSpec()
    runs: list[RunSpec] = []
    custom_env: dict[str, str] = {}
    max_generations: int = 25
    baseline: BaselineInfo = BaselineInfo()
    tools: list[ToolRef] = []


class LifecycleState(BaseModel):
    """Operational state — status, launch info, one-shot lifecycle gates."""

    model_config = ConfigDict(extra="ignore")

    status: str
    launch: LaunchInfo = LaunchInfo()
    smoke_test: SmokeTestInfo = SmokeTestInfo()
    treatment_verification: TreatmentVerificationInfo = TreatmentVerificationInfo()


class TelemetryLog(BaseModel):
    """Append-only runtime records produced during the run."""

    model_config = ConfigDict(extra="ignore")

    checkpoints: list[CheckpointEntry] = []
    mid_run_test_eval: MidRunTestEvalInfo = MidRunTestEvalInfo()
    checkpoint_analysis: CheckpointAnalysisInfo = CheckpointAnalysisInfo()
    treatment_checks: TreatmentChecksInfo = TreatmentChecksInfo()


class ControlPlane(BaseModel):
    """Live control-plane state — watchdog, notifications, cron IDs, PIDs."""

    model_config = ConfigDict(extra="ignore")

    watchdog: WatchdogSection = WatchdogSection()
    notifications: NotificationsSection = NotificationsSection()
    watchdog_pid: int | None = None
    anomaly_detector_cron_id: str | None = None
    checkpoint_cron_id: str | None = None


class ExperimentManifest(BaseModel):
    """Pydantic-validated schema for experiment.yaml (schema v2, nested storage).

    Storage is the four canonical sub-sections: ``contract``, ``lifecycle``,
    ``telemetry``, ``control_plane``. Readers access fields through these
    sub-sections — there are no flat compatibility views.

    Status-gated validation: fields that are required depend on the current
    experiment status (preregistered < implemented < running < complete).

    ``extra="ignore"`` silently drops any unknown top-level keys without
    failing validation, so task-specific metadata added outside the schema
    does not block loads.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    contract: ContractSection
    lifecycle: LifecycleState
    telemetry: TelemetryLog = TelemetryLog()
    control_plane: ControlPlane = ControlPlane()

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
        status = self.lifecycle.status
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid lifecycle.status '{status}'. "
                f"Valid statuses: {sorted(VALID_STATUSES)}"
            )
        errors: list[str] = []

        # implemented+ requires runs, servers, config, smoke_test.completed
        if status in ("implemented", "running", "complete"):
            if not self.contract.runs:
                errors.append(
                    f"contract.runs[] must be non-empty for status={status}. "
                    f"Add at least one run configuration."
                )
            config_extras = self.contract.config.extra or {}
            config_typed_set = any(
                getattr(self.contract.config, k) is not None
                for k in (
                    "problem_name",
                    "pipeline",
                    "prompt_fetcher",
                    "evolution",
                    "llm_model",
                    "n_workers",
                    "max_generations",
                )
            )
            if not config_extras and not config_typed_set:
                errors.append(
                    f"contract.config must be non-empty for status={status}. "
                    f"Add the shared Hydra config overrides."
                )
            if not self.lifecycle.smoke_test.completed:
                errors.append(
                    f"lifecycle.smoke_test.completed must be true for status={status}. "
                    f"Run a smoke test first."
                )

        # running+ requires launch info and PIDs
        if status in ("running", "complete"):
            if not self.lifecycle.launch.time:
                errors.append(
                    f"lifecycle.launch.time is required for status={status}. "
                    f"Set to the ISO timestamp of the launch."
                )
            if not self.lifecycle.launch.commit:
                errors.append(
                    f"lifecycle.launch.commit is required for status={status}. "
                    f"Set to the git commit hash at launch."
                )
            for run in self.contract.runs:
                if run.pid is None:
                    errors.append(
                        f"contract.runs[{run.label}].pid is required for status={status}. "
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
        """When control_plane.watchdog.plugin='adversarial', every run's role
        must be one of the plugin's recognized values.

        The schema keeps ``role`` as an open ``str`` so other experiment types
        (prompt_coevo, chain, optimizer, heilbron_prover, ...) can use their
        own vocabularies. Plugin-specific vocabularies are enforced here, so
        misconfiguration is caught at manifest load time rather than surfacing
        as empty population filters at runtime.
        """
        if self.control_plane.watchdog.plugin == "adversarial":
            allowed = {"constructor", "improver"}
            runs = self.contract.runs
            missing = [r.label for r in runs if r.role is None]
            if missing:
                raise ValueError(
                    f"control_plane.watchdog.plugin='adversarial' requires every "
                    f"run to set role: 'constructor' or 'improver'. Missing role "
                    f"on: {missing}"
                )
            bad = [(r.label, r.role) for r in runs if r.role not in allowed]
            if bad:
                raise ValueError(
                    f"control_plane.watchdog.plugin='adversarial' only recognizes "
                    f"roles {sorted(allowed)}. Got: {bad}"
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
        """Load and validate from a YAML file path.

        Uses OmegaConf so ``${oc.env:NAME,default}`` and cross-section
        interpolations like ``${experiment.max_generations}`` resolve at load
        time. A missing env var with no default raises loudly rather than
        leaving a literal ``${...}`` string in the manifest.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No experiment.yaml at {path}. "
                f"Create one from experiments/_template/experiment.yaml"
            )
        try:
            raw = _load_yaml_with_omegaconf(path)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Failed to load {path}: {exc}\nRecovery: git checkout {path}"
            ) from exc

        if not isinstance(raw, dict):
            raise ValueError(f"{path} must be a YAML mapping, not {type(raw).__name__}")

        try:
            return cls.from_dict(raw)
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


def _load_yaml_with_omegaconf(path: Path) -> Any:
    """Load a YAML file via OmegaConf, resolving ``${...}`` interpolations.

    Supports ``${oc.env:NAME,default}`` for env-var secrets and
    cross-section references like ``${experiment.max_generations}``.
    Missing ``${oc.env:X}`` with no default raises loudly rather than
    leaving a literal ``${...}`` string in the manifest.

    Convention for pass-through interpolations (OmegaConf-native):
    Strings meant to survive into downstream Hydra composition — e.g.
    ``post_step_hook=${composition_injection_hook}`` inside
    ``runs[].extra_overrides`` — must be escaped in YAML as
    ``post_step_hook=\\${composition_injection_hook}``. OmegaConf treats
    ``\\${...}`` as a literal dollar; the backslash is stripped during
    ``to_container`` and the string arrives in the manifest as
    ``post_step_hook=${composition_injection_hook}`` — ready for
    ``run.py``'s Hydra compose step to resolve.
    """
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def load_manifest(experiment: str) -> ExperimentManifest:
    """Load and validate experiment.yaml.

    Uses OmegaConf so ``${oc.env:NAME,default}`` and cross-section
    interpolations resolve at load time.

    Raises FileNotFoundError if the file doesn't exist.
    Raises ValueError on schema validation failure or interpolation failure.
    """
    path = manifest_path(experiment)
    if not path.exists():
        raise FileNotFoundError(
            f"No experiment.yaml at {path}. "
            f"Create one from experiments/_template/experiment.yaml"
        )

    try:
        raw = _load_yaml_with_omegaconf(path)
    except Exception as e:
        raise ValueError(
            f"Failed to load {path}: {e}\n"
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

        current = (raw.get("lifecycle") or {}).get("status", "preregistered")

        allowed = VALID_TRANSITIONS.get(current, set())
        if allow_recovery:
            allowed = allowed | RECOVERY_TRANSITIONS.get(current, set())

        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition: {current} -> {new_status}. Allowed: {allowed}"
            )

        raw.setdefault("lifecycle", {})["status"] = new_status

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
            for run in raw["contract"]["runs"]:
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
            if m.lifecycle.status in ("implemented", "running"):
                active.append(m)
        except (ValueError, yaml.YAMLError, FileNotFoundError):
            continue
    return active


def has_test_set(experiment: str) -> bool:
    """Quick check: does this experiment have a test set?"""
    m = load_manifest(experiment)
    return m.contract.problem.has_test_set


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
    identity = m.contract.identity
    lifecycle = m.lifecycle
    checkpoints = m.telemetry.checkpoints
    badge = _STATUS_BADGES.get(lifecycle.status, lifecycle.status)

    # Running badge includes generation info
    if lifecycle.status == "running" and checkpoints:
        last_cp = checkpoints[-1]
        gen = last_cp.gen
        badge = f"🟡 Running (gen {gen}/{m.contract.max_generations})"
    elif lifecycle.status == "running":
        badge = f"🟡 Running (gen 0/{m.contract.max_generations})"

    lines = [
        f"# exp: {identity.name}",
        "",
        f"**Status**: {badge}",
        f"**Branch**: `{identity.branch}`",
        f"**Tracking issue**: #{identity.tracking_issue}"
        if identity.tracking_issue
        else "",
        "",
        "## Design",
        "",
        f"See `experiments/{identity.name}/01_design.md` for full design.",
        "",
        "## Runs",
        "",
        "| Label | DB | Condition | Pipeline | PID |",
        "|-------|----|-----------|----------|-----|",
    ]

    for run in m.contract.runs:
        pid_str = str(run.pid) if run.pid else "-"
        lines.append(
            f"| {run.label} | {run.db} | {run.condition} | {run.pipeline} | {pid_str} |"
        )

    lines.extend(["", "## Checkpoints", ""])

    if checkpoints:
        lines.append("| Gen | Time | Notes |")
        lines.append("|-----|------|-------|")
        for cp in checkpoints:
            lines.append(f"| {cp.gen} | {cp.timestamp} | {cp.notes} |")
    else:
        lines.append("_No checkpoints yet._")

    baseline = m.contract.baseline
    if baseline.reference:
        lines.extend(
            [
                "",
                "## Baseline",
                "",
                f"Reference: `{baseline.reference}` "
                f"(mean={baseline.mean}, metric={baseline.metric})",
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
    "ExperimentManifest",
    "WatchdogSection",
    "AlertThresholds",
    "PlotCommand",
    # v2 sub-models (added step 3 — not yet wired into ExperimentManifest)
    "RunMetric",
    "CheckpointEntry",
    "MidRunTestEvalInfo",
    "CheckpointAnalysisEntry",
    "CheckpointAnalysisInfo",
    "CheckResult",
    "TreatmentChecksInfo",
    "ToolRef",
    "ConfigSpec",
    "PrChannelConfig",
    "TelegramChannelConfig",
    "NotificationsSection",
    # v2 sub-model groups (added step 4)
    "ExperimentIdentity",
    "ContractSection",
    "LifecycleState",
    "TelemetryLog",
    "ControlPlane",
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
