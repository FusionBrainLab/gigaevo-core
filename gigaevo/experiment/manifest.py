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
import os
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)
import yaml

from gigaevo.exceptions import ManifestValidationError
from gigaevo.experiment.lock import (
    acquire_lock,
    get_redis,
    read_manifest_rt,
    release_lock,
    write_manifest_atomic,
)


class RunRole(StrEnum):
    CONSTRUCTOR = "constructor"
    IMPROVER = "improver"


class Status(StrEnum):
    PREREGISTERED = "preregistered"
    IMPLEMENTED = "implemented"
    RUNNING = "running"
    COMPLETE = "complete"
    INVALID = "invalid"


def _resolve_proj() -> Path:
    """Resolve the project root, honoring the ``GIGAEVO_PROJ`` env override.

    Useful for tests, non-standard checkouts, and integration harnesses that
    want to point the manifest system at a scratch directory without
    monkey-patching module state.
    """
    override = os.environ.get("GIGAEVO_PROJ")
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_dir():
            raise RuntimeError(
                f"GIGAEVO_PROJ={override!r} does not point to an existing directory. "
                f"Unset it or create the directory."
            )
        return p
    return Path(__file__).parent.parent.parent


PROJ = _resolve_proj()

SUPPORTED_SCHEMA_VERSIONS = {2}

REQUIRES_IMPLEMENTATION: frozenset[Status] = frozenset(
    {
        Status.IMPLEMENTED,
        Status.RUNNING,
        Status.COMPLETE,
    }
)

REQUIRES_RUNTIME: frozenset[Status] = frozenset(
    {
        Status.RUNNING,
        Status.COMPLETE,
    }
)

TERMINAL: frozenset[Status] = frozenset(
    {
        Status.COMPLETE,
        Status.INVALID,
    }
)

VALID_TRANSITIONS: dict[Status, set[Status]] = {
    Status.PREREGISTERED: {Status.IMPLEMENTED},
    Status.IMPLEMENTED: {Status.RUNNING},
    Status.RUNNING: {Status.COMPLETE, Status.INVALID},
    Status.COMPLETE: set(),
    Status.INVALID: {Status.PREREGISTERED},
}

RECOVERY_TRANSITIONS: dict[Status, set[Status]] = {
    Status.RUNNING: {Status.IMPLEMENTED},
}

# 7 days — covers long runs + weekend without manual refresh.
DB_CLAIM_TTL_SECONDS = 7 * 24 * 60 * 60
# Backwards-compatible alias; prefer DB_CLAIM_TTL_SECONDS going forward.
DB_CLAIM_TTL = DB_CLAIM_TTL_SECONDS

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
    # Canonical event names to suppress in event_rate_zero alerts. Use when a
    # pipeline legitimately never emits a given event (e.g. simple opponent
    # provider emits no HOF_FETCH/HOF_ROTATE/CELL_PICK); listing it here stops
    # the watchdog from crying wolf without disabling the check globally.
    excluded_events: list[str] = []


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
    role: RunRole | None = None
    # Per-run pin overlay on contract.config.pinned. Dotted Hydra paths
    # (e.g. {"n_opponents": 5}) asserted by the preflight pin check (I-01,
    # I-09, I-11). Merged INTO the contract pins — a run-level key wins.
    pinned: dict[str, Any] | None = None

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
    # Per-file digests of the Hydra config tree at launch time. Written by
    # run_launch() on a fresh launch; checked by
    # _check_config_fingerprint_stable on re-launches to reject silent drift.
    config_fingerprint: dict[str, str] | None = None


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
    # `note` is frequently written as `null` in YAML (that's the default in
    # experiments/_template/experiment.yaml) which arrives here as None —
    # reject-as-string_type broke manifest validation for any manifest
    # that hadn't been touched since creation (6 redesign-sandbox yamls
    # failed this way). Accept None (coerced to "") via pre-validator so
    # downstream code can still treat `note` as a plain str.
    note: str = ""

    @field_validator("note", mode="before")
    @classmethod
    def _coerce_note_none_to_empty(cls, v: Any) -> str:
        return v if isinstance(v, str) else ""


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
    """Shared Hydra config for an experiment.

    The Hydra override hierarchy (lowest → highest precedence) is:

        1. ``task_group``            — loads ``config/experiment/<task_group>.yaml``.
                                       Active for every run in the experiment
                                       and is emitted as the FIRST Hydra override
                                       (``experiment=<task_group>``) so later
                                       args can override it.
        2. ``shared_overrides``      — key/value pairs shared by ALL runs in
                                       this experiment (e.g. ``n_opponents: 3``).
        3. ``runs[].extra_overrides`` — per-run overrides; wins over everything
                                        above (Hydra: later CLI args win).

    Standard keys are declared explicitly (``problem_name``, ``pipeline``,
    ``pinned``, …). Undeclared top-level YAML keys under ``config:`` are
    tolerated for back-compat (Pydantic ``extra="allow"``) and exposed via
    :pyattr:`flat_overrides`; new experiments should use ``shared_overrides``
    exclusively.

    The merged view used by ``launch_generator`` is :pyattr:`effective_overrides`
    (``flat_overrides`` ∪ ``shared_overrides``; nested wins on conflict).
    """

    model_config = ConfigDict(extra="allow")

    problem_name: str | None = None
    pipeline: str | None = None
    prompt_fetcher: str | None = None
    evolution: str | None = None
    llm_model: str | None = None
    n_workers: int | None = None
    # Engine-side counter is now ``max_mutants`` (post true-JIT-refresh
    # refactor); the manifest field keeps the legacy name because the value
    # also flows out unchanged to monitoring/notification UX
    # (``ExperimentUpdate.max_generations``, github_pr_channel,
    # telegram_channel) where the operator-facing "Target: N generations"
    # label is what users read. ``launch_generator`` translates this field
    # into the ``max_mutants=`` Hydra override at run launch.
    max_generations: int | None = None
    # Name of a task-group file under ``config/experiment/`` — emitted as the
    # FIRST Hydra override (``experiment=<task_group>``) so every run starts
    # from the same task-level tradition (e.g. ``heilbron`` for the Heilbron
    # triangle task). None = don't emit ``experiment=`` at all.
    task_group: str | None = None
    # Contract-level pin assertions (dotted Hydra paths → expected values).
    # Preflight fails if the resolved Hydra config drifts from these. Per-run
    # overlays live on RunSpec.pinned (I-01, I-09, I-11).
    pinned: dict[str, Any] | None = None
    # Shared Hydra overrides scoped into a nested dict. Preferred over the
    # legacy "flat model_extra" form — an explicit field makes it unambiguous
    # to downstream readers (launch_preview, checks.py, launch_generator).
    shared_overrides: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_deprecated_extra_key(cls, data: Any) -> Any:
        """Reject the old ``extra:`` YAML key with a clear migration error.

        Historical schema used ``contract.config.extra:`` as the nested
        override dict. It was renamed to ``shared_overrides`` for clarity
        (the old name collided with Pydantic's own ``extra="allow"``
        setting). Silent acceptance would hide the rename in ``model_extra``
        and break downstream consumers. Fail loudly, with the fix.
        """
        if isinstance(data, dict) and "extra" in data:
            raise ValueError(
                "contract.config.extra was renamed to contract.config.shared_overrides. "
                "Rename the YAML key — the nested dict body is unchanged."
            )
        return data

    @property
    def flat_overrides(self) -> dict[str, Any]:
        """Legacy flat top-level overrides (anything not declared as a field).

        Captured by Pydantic's ``extra="allow"`` into ``model_extra``. Kept
        only for back-compat with pre-``shared_overrides`` manifests — new
        experiments should put all cross-run overrides under
        :pyattr:`shared_overrides`.
        """
        return dict(self.model_extra or {})

    @property
    def effective_overrides(self) -> dict[str, Any]:
        """Merged view of ``flat_overrides`` ∪ ``shared_overrides``.

        Nested ``shared_overrides`` wins on key collision. Use this as the
        single source of truth when enumerating Hydra overrides shared
        across runs — ``launch_generator`` and ``launch_preview`` both read
        from here.
        """
        return {**self.flat_overrides, **(self.shared_overrides or {})}


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
    servers: list[str] = []
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
    experiment status. Forward transitions: preregistered → implemented → running
    → {complete | invalid}. Recovery edge: running → implemented via recover_status.

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
        status_str = self.lifecycle.status
        try:
            status = Status(status_str)
        except ValueError:
            valid = ", ".join(s.value for s in Status)
            raise ValueError(
                f"Invalid lifecycle.status '{status_str}'. Valid statuses: {valid}"
            ) from None

        errors: list[str] = []

        # implemented+ requires runs, servers, config, smoke_test.completed
        if status in REQUIRES_IMPLEMENTATION:
            if not self.contract.runs:
                errors.append(
                    f"contract.runs[] must be non-empty for status={status.value}. "
                    f"Add at least one run configuration."
                )
            if not self.contract.servers:
                errors.append(
                    f"contract.servers[] must be non-empty for status={status.value}. "
                    f"Add the server hostnames used by this experiment."
                )
            config_extras = self.contract.config.effective_overrides
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
                    f"contract.config must be non-empty for status={status.value}. "
                    f"Add the shared Hydra config overrides."
                )
            if not self.lifecycle.smoke_test.completed:
                errors.append(
                    f"lifecycle.smoke_test.completed must be true for status={status.value}. "
                    f"Run a smoke test first."
                )

        # running+ requires launch info and PIDs
        if status in REQUIRES_RUNTIME:
            if not self.lifecycle.launch.time:
                errors.append(
                    f"lifecycle.launch.time is required for status={status.value}. "
                    f"Set to the ISO timestamp of the launch."
                )
            if not self.lifecycle.launch.commit:
                errors.append(
                    f"lifecycle.launch.commit is required for status={status.value}. "
                    f"Set to the git commit hash at launch."
                )
            for run in self.contract.runs:
                if run.pid is None:
                    errors.append(
                        f"contract.runs[{run.label}].pid is required for status={status.value}. "
                        f"Record the PID after launching."
                    )

        if errors:
            raise ValueError(
                f"Manifest validation failed (status={status.value}):\n"
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
        raw = _read_yaml_file(path)
        try:
            return cls.from_dict(raw)
        except (PydanticValidationError, ValueError) as exc:
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


def _read_yaml_file(path: Path) -> dict[str, Any]:
    """Read and parse experiment.yaml via OmegaConf.

    Shared IO path for ``load_manifest`` and ``ExperimentManifest.from_yaml_file``.
    Raises ``FileNotFoundError`` for missing files and ``ValueError`` for
    unreadable/non-mapping YAML — never a bare ``Exception``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No experiment.yaml at {path}. "
            f"Create one from experiments/_template/experiment.yaml"
        )
    try:
        raw = _load_yaml_with_omegaconf(path)
    except (yaml.YAMLError, OSError, ValueError) as exc:
        raise ValueError(
            f"Failed to load {path}: {exc}\nRecovery: git checkout {path}"
        ) from exc

    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a YAML mapping, not {type(raw).__name__}")
    return raw


_KNOWN_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "contract", "lifecycle", "telemetry", "control_plane"}
)


def load_manifest(experiment: str, *, strict: bool = False) -> ExperimentManifest:
    """Load and validate ``experiment.yaml``.

    Uses OmegaConf so ``${oc.env:NAME,default}`` and cross-section
    interpolations resolve at load time.

    When ``strict=True``, reject any unknown top-level keys. This catches
    typos at the v2 root (``lifeycycle:`` vs ``lifecycle:``) that would
    otherwise be silently dropped by the model's ``extra="ignore"`` policy.
    Sub-section extras are by design (``ConfigSpec`` Hydra knobs,
    ``RunMetric`` problem-specific ``best_X`` metrics) and remain allowed.

    Raises FileNotFoundError if the file doesn't exist.
    Raises ManifestValidationError on schema validation failure or
        (in strict mode) on unknown top-level keys.
    Raises ValueError on YAML parse / interpolation failure.
    """
    path = manifest_path(experiment)
    raw = _read_yaml_file(path)
    if strict:
        unknown = sorted(set(raw) - _KNOWN_TOP_LEVEL_KEYS)
        if unknown:
            raise ManifestValidationError(
                experiment,
                f"unknown top-level keys: {unknown}. "
                f"Allowed: {sorted(_KNOWN_TOP_LEVEL_KEYS)}",
            )
    return _validate(raw, experiment)


def _validate(raw: dict[str, Any], experiment: str) -> ExperimentManifest:
    """Validate raw YAML dict and return ExperimentManifest.

    Wraps Pydantic ``ValidationError`` in ``ManifestValidationError`` so
    downstream callers can recognize manifest-shape failures (as opposed
    to IO or YAML parsing errors) without inspecting exception messages.
    """
    try:
        return ExperimentManifest.from_dict(raw)
    except (PydanticValidationError, ValueError) as exc:
        raise ManifestValidationError(experiment, str(exc)) from exc


# ---------------------------------------------------------------------------
# Status Transitions
# ---------------------------------------------------------------------------


def set_status(
    experiment: str,
    new_status: str,
) -> ExperimentManifest:
    """Forward-only status transition. Validates state machine.

    Args:
        experiment: e.g. "hover/feedback_softfit"
        new_status: target status (must be reachable via VALID_TRANSITIONS)

    Returns:
        Updated manifest.

    Raises:
        ValueError on invalid transition or validation failure.
    """
    r = get_redis()
    lock_key = acquire_lock(r, experiment)
    try:
        path = manifest_path(experiment)
        raw = read_manifest_rt(path)

        current = (raw.get("lifecycle") or {}).get("status", "preregistered")
        try:
            current_status = Status(current)
        except ValueError:
            raise ValueError(f"Invalid current status: {current}") from None

        allowed = VALID_TRANSITIONS.get(current_status, set())
        target_status = Status(new_status)

        if target_status not in allowed:
            raise ValueError(
                f"Invalid transition: {current} -> {new_status}. Allowed: {', '.join(s.value for s in allowed)}"
            )

        raw.setdefault("lifecycle", {})["status"] = new_status

        manifest = _validate(raw, experiment)

        write_manifest_atomic(path, raw)
        return manifest
    finally:
        release_lock(r, lock_key)


def recover_status(
    experiment: str,
    new_status: str,
) -> ExperimentManifest:
    """Recovery-only status transition. For running → implemented recovery.

    Allows transitions defined in RECOVERY_TRANSITIONS (currently: running → implemented).
    Used when rolling back a stuck experiment to retry from an earlier state.

    Args:
        experiment: e.g. "hover/feedback_softfit"
        new_status: target status (must be reachable via RECOVERY_TRANSITIONS)

    Returns:
        Updated manifest.

    Raises:
        ValueError on invalid recovery transition or validation failure.
    """
    r = get_redis()
    lock_key = acquire_lock(r, experiment)
    try:
        path = manifest_path(experiment)
        raw = read_manifest_rt(path)

        current = (raw.get("lifecycle") or {}).get("status", "preregistered")
        try:
            current_status = Status(current)
        except ValueError:
            raise ValueError(f"Invalid current status: {current}") from None

        allowed = RECOVERY_TRANSITIONS.get(current_status, set())
        target_status = Status(new_status)

        if target_status not in allowed:
            raise ValueError(
                f"Cannot recover to {new_status} from {current}. "
                f"Recovery transitions: {', '.join(f'{k.value} → {", ".join(s.value for s in v)}' for k, v in RECOVERY_TRANSITIONS.items())}"
            )

        raw.setdefault("lifecycle", {})["status"] = new_status

        manifest = _validate(raw, experiment)
        write_manifest_atomic(path, raw)
        return manifest
    finally:
        release_lock(r, lock_key)


def update_manifest(
    experiment: str,
    updater: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> ExperimentManifest:
    """Read-modify-write ``experiment.yaml`` under a Redis lock.

    The updater receives the parsed manifest as a round-trip
    :class:`ruamel.yaml.comments.CommentedMap` and may either:

    - **mutate it in-place** and return ``None`` (preferred — preserves
      comments and key order on disk), or
    - **return a new mapping** that replaces it entirely (loses any
      comments not present on the new mapping).

    In both cases validation runs against the post-update mapping before
    the atomic write.

    Usage::

        def set_pids(raw):
            for run in raw["contract"]["runs"]:
                if run["label"] == "F1":
                    run["pid"] = 12345

        update_manifest("hover/feedback_softfit", set_pids)
    """
    r = get_redis()
    lock_key = acquire_lock(r, experiment)
    try:
        path = manifest_path(experiment)
        # Widened type so the optional dict-replacement branch type-checks;
        # CommentedMap is a MutableMapping but isn't a dict subclass.
        raw: dict[str, Any] = read_manifest_rt(path)

        result = updater(raw)
        if result is not None:
            raw = result

        manifest = _validate(raw, experiment)
        write_manifest_atomic(path, raw)
        return manifest
    finally:
        release_lock(r, lock_key)


# ---------------------------------------------------------------------------
# DB Claims
# ---------------------------------------------------------------------------


def _db_claim_key(db: int) -> str:
    return f"experiments:db_claim:{db}"


# Lua: atomic all-or-nothing claim.
# KEYS: N db_claim keys. ARGV[1]=experiment, ARGV[2]=TTL, ARGV[3..]=db numbers.
# Returns a flat list [db1, owner1, db2, owner2, ...] of conflicts. An empty
# list means every claim succeeded (new or idempotent re-claim by same owner).
# On any conflict, NO keys are written.
_CLAIM_DBS_LUA = """
local failed = {}
local exp = ARGV[1]
local ttl = ARGV[2]
for i=1,#KEYS do
  local cur = redis.call('GET', KEYS[i])
  if cur and cur ~= exp then
    table.insert(failed, ARGV[2+i])
    table.insert(failed, cur)
  end
end
if #failed > 0 then
  return failed
end
for i=1,#KEYS do
  redis.call('SET', KEYS[i], exp, 'EX', ttl)
end
return {}
"""

# Lua: CAS refresh — extend TTL only if the current owner matches.
# Returns number of keys refreshed. Silently skips keys owned by others or
# already expired; caller can compare against len(dbs) to detect drift.
_REFRESH_DB_CLAIMS_LUA = """
local refreshed = 0
local exp = ARGV[1]
local ttl = ARGV[2]
for i=1,#KEYS do
  if redis.call('GET', KEYS[i]) == exp then
    redis.call('EXPIRE', KEYS[i], ttl)
    refreshed = refreshed + 1
  end
end
return refreshed
"""

# Lua: CAS release — delete only if the current owner matches.
# Prevents experiment A from releasing experiment B's claim on the same DB.
_RELEASE_DB_CLAIMS_LUA = """
local released = 0
local exp = ARGV[1]
for i=1,#KEYS do
  if redis.call('GET', KEYS[i]) == exp then
    redis.call('DEL', KEYS[i])
    released = released + 1
  end
end
return released
"""


def claim_dbs(experiment: str, dbs: list[int]) -> list[tuple[int, str]]:
    """Atomically claim Redis DBs — all or nothing.

    If any requested DB is currently owned by a different experiment, NO
    claims are written and the conflict list is returned as
    ``[(db, owner), ...]``. Re-claiming a DB already owned by the same
    experiment is idempotent (TTL is not refreshed — use
    :func:`refresh_db_claims` for that).

    Uses a server-side Lua script so the check-then-set is atomic across
    all DBs; there is no window where a half-claimed state is visible.
    """
    if not dbs:
        return []
    r = get_redis()
    keys = [_db_claim_key(db) for db in dbs]
    args = [experiment, str(DB_CLAIM_TTL_SECONDS), *[str(db) for db in dbs]]
    result = r.eval(_CLAIM_DBS_LUA, len(keys), *keys, *args)
    if not result:
        return []
    failed: list[tuple[int, str]] = []
    for i in range(0, len(result), 2):
        db = int(result[i].decode() if isinstance(result[i], bytes) else result[i])
        owner = (
            result[i + 1].decode()
            if isinstance(result[i + 1], bytes)
            else str(result[i + 1])
        )
        failed.append((db, owner))
    return failed


def refresh_db_claims(experiment: str, dbs: list[int]) -> int:
    """Refresh TTL on already-claimed DBs (called by watchdog each cycle).

    Owner-checked (CAS): a claim is refreshed only when its current owner
    matches ``experiment``. Returns the number of keys successfully
    refreshed — callers can compare against ``len(dbs)`` to detect
    claim drift (keys that expired or got stolen).
    """
    if not dbs:
        return 0
    r = get_redis()
    keys = [_db_claim_key(db) for db in dbs]
    return int(
        r.eval(
            _REFRESH_DB_CLAIMS_LUA,
            len(keys),
            *keys,
            experiment,
            str(DB_CLAIM_TTL_SECONDS),
        )
    )


def release_db_claims(experiment: str, dbs: list[int]) -> int:
    """Release DB claims (called by reset_status and closeout).

    Owner-checked (CAS): a claim is released only when its current owner
    matches ``experiment``. Prevents experiment A from accidentally
    releasing experiment B's claim on a shared DB number. Returns the
    number of claims actually released.
    """
    if not dbs:
        return 0
    r = get_redis()
    keys = [_db_claim_key(db) for db in dbs]
    return int(r.eval(_RELEASE_DB_CLAIMS_LUA, len(keys), *keys, experiment))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_active_experiments() -> list[ExperimentManifest]:
    """Scan for experiments with status in (implemented, running).

    Includes 'implemented' to prevent TOCTOU race with DB claims.

    Silently skipping broken manifests has historically masked real bugs
    (stale PROJ root, malformed YAML, schema drift). We still *continue*
    past a broken manifest — otherwise one stale yaml would block all
    launches — but each skip is logged at WARNING so it shows up in CI.
    """
    active: list[ExperimentManifest] = []
    experiments_dir = PROJ / "experiments"
    for yaml_path in experiments_dir.glob("*/*/experiment.yaml"):
        if "_template" in str(yaml_path):
            continue
        rel = yaml_path.parent.relative_to(experiments_dir)
        experiment = str(rel)
        try:
            m = load_manifest(experiment)
        except FileNotFoundError:
            continue
        except (ManifestValidationError, ValueError, yaml.YAMLError) as exc:
            logger.warning(
                "find_active_experiments: skipping {} — {}: {}",
                experiment,
                type(exc).__name__,
                exc,
            )
            continue
        if m.lifecycle.status in ("implemented", "running"):
            active.append(m)
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


def _format_status_badge(
    status: str, max_generations: int, last_checkpoint_gen: int | None
) -> str:
    """Pick the badge string for a given lifecycle status.

    The ``running`` badge is overlaid with current/total generations so the
    PR title reflects live progress; everything else is a static lookup.
    """
    if status != "running":
        return _STATUS_BADGES.get(status, status)
    gen = last_checkpoint_gen if last_checkpoint_gen is not None else 0
    return f"🟡 Running (gen {gen}/{max_generations})"


def _render_header(identity: ExperimentIdentity, badge: str) -> list[str]:
    lines = [
        f"# exp: {identity.name}",
        "",
        f"**Status**: {badge}",
        f"**Branch**: `{identity.branch}`",
    ]
    if identity.tracking_issue:
        lines.append(f"**Tracking issue**: #{identity.tracking_issue}")
    return lines


def _render_design_link(name: str) -> list[str]:
    return [
        "",
        "## Design",
        "",
        f"See `experiments/{name}/01_design.md` for full design.",
    ]


def _render_runs_table(runs: list[RunSpec]) -> list[str]:
    lines = [
        "",
        "## Runs",
        "",
        "| Label | DB | Condition | Pipeline | PID |",
        "|-------|----|-----------|----------|-----|",
    ]
    for run in runs:
        pid_str = str(run.pid) if run.pid else "-"
        lines.append(
            f"| {run.label} | {run.db} | {run.condition} | {run.pipeline} | {pid_str} |"
        )
    return lines


def _render_checkpoints(checkpoints: list[CheckpointEntry]) -> list[str]:
    lines = ["", "## Checkpoints", ""]
    if not checkpoints:
        lines.append("_No checkpoints yet._")
        return lines
    lines.append("| Gen | Time | Notes |")
    lines.append("|-----|------|-------|")
    for cp in checkpoints:
        lines.append(f"| {cp.gen} | {cp.timestamp} | {cp.notes} |")
    return lines


def _render_baseline(baseline: BaselineInfo) -> list[str]:
    if not baseline.reference:
        return []
    return [
        "",
        "## Baseline",
        "",
        f"Reference: `{baseline.reference}` "
        f"(mean={baseline.mean}, metric={baseline.metric})",
    ]


def generate_pr_description(experiment: str) -> str:
    """Render ``PR_DESCRIPTION.md`` content from the experiment manifest."""
    m = load_manifest(experiment)
    last_cp_gen = m.telemetry.checkpoints[-1].gen if m.telemetry.checkpoints else None
    badge = _format_status_badge(
        m.lifecycle.status, m.contract.max_generations, last_cp_gen
    )

    sections: list[str] = []
    sections += _render_header(m.contract.identity, badge)
    sections += _render_design_link(m.contract.identity.name)
    sections += _render_runs_table(m.contract.runs)
    sections += _render_checkpoints(m.telemetry.checkpoints)
    sections += _render_baseline(m.contract.baseline)
    sections += ["", "## Archives", "", "_(pending)_", ""]
    return "\n".join(sections) + "\n"


__all__ = [
    "ManifestValidationError",
    "Status",
    "REQUIRES_IMPLEMENTATION",
    "REQUIRES_RUNTIME",
    "TERMINAL",
    "VALID_TRANSITIONS",
    "RECOVERY_TRANSITIONS",
    "DB_CLAIM_TTL",
    "DB_CLAIM_TTL_SECONDS",
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
    "recover_status",
    "update_manifest",
    "claim_dbs",
    "refresh_db_claims",
    "release_db_claims",
    "find_active_experiments",
    "has_test_set",
    "generate_pr_description",
    # Re-exports from gigaevo.experiment.lock for callers that already
    # import from manifest. Prefer importing directly from the lock module
    # for new code.
    "get_redis",
    "acquire_lock",
    "release_lock",
    "write_manifest_atomic",
]
