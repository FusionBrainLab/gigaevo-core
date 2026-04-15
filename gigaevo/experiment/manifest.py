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

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
import yaml

from gigaevo.experiment.lock import (
    _acquire_lock,
    _get_redis,
    _release_lock,
    _write_manifest_atomic,
)

PROJ = Path(__file__).parent.parent.parent

SUPPORTED_SCHEMA_VERSIONS = {1, 2}
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
    role: str | None = None

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
    """Launch state information.

    ``extra="allow"`` preserves v1 sidecar fields (``anomaly_detector_cron_id``,
    ``checkpoint_cron_id``) that the v2 schema relocates under ``control_plane``.
    Keeping them round-trippable here lets ``ExperimentManifest.control_plane``
    surface them for in-memory reads until the step-7 yaml migration lands.
    """

    model_config = ConfigDict(extra="allow")

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


# ---------------------------------------------------------------------------
# New typed sub-models for schema v2 (added additively in step 3).
#
# These are not yet wired into ExperimentManifest — step 4 will introduce the
# ContractSection / LifecycleState / TelemetryLog / ControlPlane groups that
# use them. For now they exist as first-class, round-trippable types so
# downstream code (migration script, preflight, checkpoint skill) can start
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
    """Target v2 shape for treatment-check verification state.

    The v1 yamls store this field in multiple shapes (list of checks, dict of
    pattern lists, ...). The migration script in step 6 normalizes them into
    this single shape.
    """

    model_config = ConfigDict(extra="ignore")

    completed: bool = False
    completed_at: str | None = None
    results: list[CheckResult] = []


class StopCondition(BaseModel):
    """Structured component of ``StoppingRule.conditions``."""

    model_config = ConfigDict(extra="ignore")

    kind: Literal["invalidity_window", "fitness_plateau", "manual"]
    threshold: float | None = None
    window: int | None = None
    note: str = ""


class StoppingRule(BaseModel):
    """Structured stopping rule — replaces the unenforced prose string in v2.

    ``description`` preserves the pre-registered prose; ``conditions`` holds the
    machine-checkable variant; ``enforce_at`` selects whether the engine, the
    checkpoint skill, or neither enforces the rule. Engine enforcement is
    deferred to a follow-up PR (see plan §Stopping Rule).
    """

    model_config = ConfigDict(extra="ignore")

    description: str = ""
    conditions: list[StopCondition] = []
    enforce_at: Literal["engine", "checkpoint", "none"] = "checkpoint"


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
# Schema v2 top-level sub-model groups (step 4).
#
# These bundle the existing flat fields under four concerns:
#   contract      — pre-registered, researcher-authored, frozen at preregistered
#   lifecycle     — operational state (status, launch, smoke test, treatment verify)
#   telemetry     — append-only runtime records (checkpoints, analyses, checks)
#   control_plane — live processes + dashboards (watchdog, notifications, crons)
#
# For v1 inputs (flat yaml), ExperimentManifest exposes these as computed views
# derived from the flat fields. Step 5 switches the loader so v2 (nested) yamls
# become the canonical on-disk shape; at that point flat fields are derived
# from the nested sub-groups (and finally removed in step 9).
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
    stopping_rule: StoppingRule = StoppingRule()
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

    ``extra="allow"`` is required for v1 compatibility: v1 yamls carry
    ``mid_run_test_eval``, ``checkpoint_analysis`` and ``treatment_checks``
    at the top level. The v2 schema relocates them under ``telemetry``;
    until the step-7 yaml migration lands we keep them round-trippable here
    so the computed ``telemetry`` view can surface them.
    """

    model_config = ConfigDict(extra="allow")

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
        """When watchdog.plugin='adversarial', every run's role must be one of
        the plugin's recognized values.

        The schema keeps ``role`` as an open ``str`` so other experiment types
        (prompt_coevo, chain, optimizer, heilbron_prover, ...) can use their
        own vocabularies. Plugin-specific vocabularies are enforced here, so
        misconfiguration is caught at manifest load time rather than surfacing
        as empty population filters at runtime.
        """
        if self.watchdog.plugin == "adversarial":
            allowed = {"constructor", "improver"}
            missing = [r.label for r in self.runs if r.role is None]
            if missing:
                raise ValueError(
                    f"watchdog.plugin='adversarial' requires every run to set "
                    f"role: 'constructor' or 'improver'. Missing role on: "
                    f"{missing}"
                )
            bad = [(r.label, r.role) for r in self.runs if r.role not in allowed]
            if bad:
                raise ValueError(
                    f"watchdog.plugin='adversarial' only recognizes roles "
                    f"{sorted(allowed)}. Got: {bad}"
                )
        return self

    # ---- schema v2 view properties (step 4) ------------------------------
    #
    # These derive the four target sub-groups from the current flat fields so
    # downstream code can start consuming ``m.contract.*`` / ``m.lifecycle.*``
    # / ``m.telemetry.*`` / ``m.control_plane.*`` while the on-disk yamls
    # remain v1. Step 5 (OmegaConf loader) and step 7 (yaml migration) flip
    # the primary direction so nested becomes canonical.

    @property
    def contract(self) -> ContractSection:
        """ContractSection view derived from flat fields."""
        return ContractSection(
            identity=ExperimentIdentity(
                name=self.experiment.name,
                task=self.experiment.task,
                branch=self.experiment.branch,
                prereg_commit=self.experiment.prereg_commit,
                pr_number=self.experiment.pr_number,
                tracking_issue=self.experiment.tracking_issue,
            ),
            problem=self.problem,
            config=ConfigSpec(extra=dict(self.config)) if self.config else ConfigSpec(),
            runs=list(self.runs),
            servers=list(self.servers),
            custom_env=dict(self.custom_env),
            max_generations=self.experiment.max_generations,
            stopping_rule=StoppingRule(description=self.experiment.stopping_rule or ""),
            baseline=self.baseline,
            tools=[
                ToolRef.model_validate(t) if isinstance(t, dict) else t
                for t in self.tools
            ],
        )

    @property
    def lifecycle(self) -> LifecycleState:
        """LifecycleState view derived from flat fields."""
        return LifecycleState(
            status=self.experiment.status,
            launch=self.launch,
            smoke_test=self.smoke_test,
            treatment_verification=self.treatment_verification,
        )

    @property
    def telemetry(self) -> TelemetryLog:
        """TelemetryLog view derived from flat fields + raw extras."""
        extras = self.model_extra or {}
        mre_raw = extras.get("mid_run_test_eval")
        ca_raw = extras.get("checkpoint_analysis")
        tc_raw = extras.get("treatment_checks")

        mid_run = (
            MidRunTestEvalInfo.model_validate(mre_raw)
            if isinstance(mre_raw, dict)
            else MidRunTestEvalInfo()
        )
        ca = (
            CheckpointAnalysisInfo.model_validate(ca_raw)
            if isinstance(ca_raw, dict)
            else CheckpointAnalysisInfo()
        )
        # v1 ``treatment_checks`` has two shapes (list of checks vs dict of
        # pattern lists) — neither matches the v2 target shape. Normalize by
        # returning the default; the step 6 migration CLI handles conversion.
        tc = (
            TreatmentChecksInfo.model_validate(tc_raw)
            if isinstance(tc_raw, dict)
            and "results" in tc_raw
            and isinstance(tc_raw.get("results"), list)
            else TreatmentChecksInfo()
        )

        checkpoints = [
            CheckpointEntry.model_validate(cp) if isinstance(cp, dict) else cp
            for cp in self.checkpoints
        ]

        return TelemetryLog(
            checkpoints=checkpoints,
            mid_run_test_eval=mid_run,
            checkpoint_analysis=ca,
            treatment_checks=tc,
        )

    @property
    def control_plane(self) -> ControlPlane:
        """ControlPlane view derived from flat fields + LaunchInfo extras."""
        launch_extras = self.launch.model_extra or {}
        return ControlPlane(
            watchdog=self.watchdog,
            notifications=NotificationsSection(),  # v1 has no notifications — defaults
            watchdog_pid=self.launch.watchdog_pid,
            anomaly_detector_cron_id=launch_extras.get("anomaly_detector_cron_id"),
            checkpoint_cron_id=launch_extras.get("checkpoint_cron_id"),
        )

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
# v1 → v2 migration (pure dict transform used by step 6 migration CLI)
# ---------------------------------------------------------------------------


def _migrate_v1_to_v2(raw: dict[str, Any]) -> dict[str, Any]:
    """Transform a flat v1 manifest dict into a nested v2 dict.

    Input (v1):   top-level ``experiment``, ``runs``, ``watchdog``, ``launch``,
                  ``checkpoints``, etc. scattered across the root.
    Output (v2):  four sub-sections ``contract`` / ``lifecycle`` / ``telemetry``
                  / ``control_plane`` with the same content re-homed.

    This is a pure function — the input dict is not mutated. The output must
    validate against ``ExperimentManifest`` (which accepts both shapes during
    the transition).
    """
    import copy

    src = copy.deepcopy(raw)

    exp = src.get("experiment", {}) or {}
    problem = src.get("problem", {}) or {}
    runs = src.get("runs", []) or []
    servers = src.get("servers", []) or []
    config_raw = src.get("config", {}) or {}
    custom_env = src.get("custom_env", {}) or {}
    baseline = src.get("baseline", {}) or {}
    tools = src.get("tools", []) or []
    launch = dict(src.get("launch", {}) or {})
    smoke_test = src.get("smoke_test", {}) or {}
    treatment_verification = src.get("treatment_verification", {}) or {}
    watchdog = src.get("watchdog", {}) or {}
    checkpoints = src.get("checkpoints", []) or []
    mid_run_test_eval = src.get("mid_run_test_eval", {}) or {}
    checkpoint_analysis = src.get("checkpoint_analysis", {}) or {}
    treatment_checks_raw = src.get("treatment_checks")

    # contract.identity — pull identity fields off of ``experiment.*``
    identity = {
        "name": exp.get("name"),
        "task": exp.get("task"),
        "branch": exp.get("branch", ""),
        "prereg_commit": exp.get("prereg_commit") or src.get("prereg_commit"),
        "pr_number": exp.get("pr_number") or src.get("pr_number"),
        "tracking_issue": exp.get("tracking_issue") or src.get("tracking_issue"),
    }

    # contract.config — v1 is a flat dict; v2 nests problem-specific knobs
    # under ``extra`` so the typed keys stay separate.
    typed_cfg_keys = {
        "problem_name",
        "pipeline",
        "prompt_fetcher",
        "evolution",
        "llm_model",
        "n_workers",
        "max_generations",
    }
    cfg_typed = {k: config_raw[k] for k in typed_cfg_keys if k in config_raw}
    cfg_extra = {k: v for k, v in config_raw.items() if k not in typed_cfg_keys}
    config_v2: dict[str, Any] = dict(cfg_typed)
    config_v2["extra"] = cfg_extra

    # contract.stopping_rule — prose in v1 becomes StoppingRule.description
    stopping_rule_v2 = {
        "description": exp.get("stopping_rule", "") or "",
        "conditions": [],
    }

    contract = {
        "identity": identity,
        "problem": problem,
        "config": config_v2,
        "runs": runs,
        "servers": servers,
        "custom_env": custom_env,
        "max_generations": exp.get("max_generations", 25),
        "stopping_rule": stopping_rule_v2,
        "baseline": baseline,
        "tools": tools,
    }

    # control_plane sidecars (cron IDs + watchdog PID) are pulled off launch.*
    watchdog_pid = launch.pop("watchdog_pid", None)
    anomaly_cron = launch.pop("anomaly_detector_cron_id", None)
    checkpoint_cron = launch.pop("checkpoint_cron_id", None)

    lifecycle = {
        "status": exp.get("status", "preregistered"),
        "launch": launch,
        "smoke_test": smoke_test,
        "treatment_verification": treatment_verification,
    }

    # telemetry.treatment_checks — v1 has two incompatible shapes. Only carry
    # forward when the v1 shape already matches the v2 target (list-of-dicts
    # under a ``results`` key). Otherwise emit an empty default; operators can
    # hand-populate after migration.
    if (
        isinstance(treatment_checks_raw, dict)
        and "results" in treatment_checks_raw
        and isinstance(treatment_checks_raw.get("results"), list)
    ):
        treatment_checks_v2 = treatment_checks_raw
    else:
        treatment_checks_v2 = {"completed": False, "results": []}

    telemetry = {
        "checkpoints": checkpoints,
        "mid_run_test_eval": mid_run_test_eval,
        "checkpoint_analysis": checkpoint_analysis,
        "treatment_checks": treatment_checks_v2,
    }

    control_plane = {
        "watchdog": watchdog,
        "notifications": {
            "pr": {"enabled": True, "comment_mode": "rolling"},
            "telegram": {
                "enabled": True,
                "chat_id_env": "TELEGRAM_CHAT_ID",
                "token_env": "TELEGRAM_BOT_TOKEN",
            },
        },
        "watchdog_pid": watchdog_pid,
        "anomaly_detector_cron_id": anomaly_cron,
        "checkpoint_cron_id": checkpoint_cron,
    }

    return {
        "schema_version": 2,
        "experiment": exp,  # keep flat ExperimentSection for backward-compat reads
        "problem": problem,
        "runs": runs,
        "servers": servers,
        "config": config_raw,
        "custom_env": custom_env,
        "baseline": baseline,
        "smoke_test": smoke_test,
        "launch": launch,
        "treatment_verification": treatment_verification,
        "watchdog": watchdog,
        "checkpoints": checkpoints,
        "tools": tools,
        "contract": contract,
        "lifecycle": lifecycle,
        "telemetry": telemetry,
        "control_plane": control_plane,
    }


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
    # v2 sub-models (added step 3 — not yet wired into ExperimentManifest)
    "RunMetric",
    "CheckpointEntry",
    "MidRunTestEvalInfo",
    "CheckpointAnalysisEntry",
    "CheckpointAnalysisInfo",
    "CheckResult",
    "TreatmentChecksInfo",
    "StopCondition",
    "StoppingRule",
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
    "SUPPORTED_SCHEMA_VERSIONS",
    "_migrate_v1_to_v2",
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
