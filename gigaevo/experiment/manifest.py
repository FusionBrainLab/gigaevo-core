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

# Top-level flat keys produced by v1 yamls (and by v2 intermediate yamls that
# still carry duplicated flat+nested sections via YAML anchors). The ``before``
# validator synthesizes the canonical nested sections from these, then strips
# them so ``extra="allow"`` doesn't also round-trip the flat shape. Any key
# added here must also be covered in ``tools/experiment/flatten_manifest_v2.py``.
_LEGACY_FLAT_KEYS = frozenset(
    {
        "experiment",
        "problem",
        "runs",
        "servers",
        "config",
        "custom_env",
        "baseline",
        "tools",
        "stopping_rule",
        "launch",
        "smoke_test",
        "treatment_verification",
        "watchdog",
        "checkpoints",
        "mid_run_test_eval",
        "checkpoint_analysis",
        "treatment_checks",
    }
)


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
    """Pydantic-validated schema for experiment.yaml (schema v2, nested storage).

    Storage is the four canonical sub-sections: ``contract``, ``lifecycle``,
    ``telemetry``, ``control_plane``. A ``model_validator(mode="before")``
    reshuffles legacy flat-shaped v1 yamls into nested form so older files
    keep loading during the transition.

    Flat compatibility views (``experiment``, ``runs``, ``servers``, ``problem``,
    ``config``, ``custom_env``, ``baseline``, ``smoke_test``, ``tools``,
    ``launch``, ``watchdog``, ``checkpoints``, ``treatment_verification``) are
    exposed as computed properties that derive from the nested storage. They
    exist only to let legacy readers keep working; new code should read the
    nested sub-sections directly.

    Status-gated validation: fields that are required depend on the current
    experiment status (preregistered < implemented < running < complete).

    ``extra="allow"`` absorbs any legacy top-level keys that survive the
    normalizing step (e.g. unknown task-specific metadata) without failing
    validation.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int
    contract: ContractSection
    lifecycle: LifecycleState
    telemetry: TelemetryLog = TelemetryLog()
    control_plane: ControlPlane = ControlPlane()

    # ---- legacy-shape normalizer ----------------------------------------
    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_shape(cls, data: Any) -> Any:
        """Accept both nested-only (v2 canonical) and flat-with-nested (v2
        intermediate) and legacy flat-only v1 yamls.

        When the caller provides only flat v1 keys (no ``contract`` /
        ``lifecycle`` sections), build the nested sub-sections from them. When
        both shapes are present, the nested sections win — they are the
        canonical source of truth.
        """
        if not isinstance(data, dict):
            return data

        # If the canonical nested sections are already present, trust them.
        has_contract = isinstance(data.get("contract"), dict)
        has_lifecycle = isinstance(data.get("lifecycle"), dict)
        if has_contract and has_lifecycle:
            # Strip duplicated flat keys so the model doesn't also see them
            # via ``extra="allow"`` (which would cause confusion in tests
            # that inspect ``model_extra``).
            for k in _LEGACY_FLAT_KEYS:
                data.pop(k, None)
            return data

        # Legacy flat-shaped v1 yaml: synthesize nested sections.
        flat_experiment = data.get("experiment") or {}
        flat_runs = data.get("runs") or []
        flat_servers = data.get("servers") or []
        flat_config = data.get("config") or {}
        flat_custom_env = data.get("custom_env") or {}
        flat_problem = data.get("problem") or {}
        flat_baseline = data.get("baseline") or {}
        flat_tools = data.get("tools") or []
        flat_launch = data.get("launch") or {}
        flat_smoke = data.get("smoke_test") or {}
        flat_tv = data.get("treatment_verification") or {}
        flat_watchdog = data.get("watchdog") or {}
        flat_checkpoints = data.get("checkpoints") or []
        flat_mid_run = data.get("mid_run_test_eval") or {}
        flat_ca = data.get("checkpoint_analysis") or {}
        flat_tc = data.get("treatment_checks") or {}
        flat_stopping_rule = data.get("stopping_rule")

        sr_prose = flat_experiment.get("stopping_rule", "") or ""
        if isinstance(flat_stopping_rule, dict):
            stopping_rule = flat_stopping_rule
        else:
            stopping_rule = {"description": sr_prose, "conditions": []}

        # Normalize config: flat is a raw dict; contract.config is a typed
        # ConfigSpec with most keys under ``extra``.
        config_block: dict[str, Any] = {"extra": dict(flat_config)}
        for k in (
            "problem_name",
            "pipeline",
            "prompt_fetcher",
            "evolution",
            "llm_model",
            "n_workers",
            "max_generations",
        ):
            if k in flat_config:
                config_block[k] = flat_config[k]
                config_block["extra"].pop(k, None)

        # Build identity block, letting Pydantic catch missing required name/task
        # rather than defaulting them to empty strings.
        identity_block: dict[str, Any] = {}
        if "name" in flat_experiment:
            identity_block["name"] = flat_experiment["name"]
        if "task" in flat_experiment:
            identity_block["task"] = flat_experiment["task"]
        identity_block.update({
            "branch": flat_experiment.get("branch", ""),
            "prereg_commit": flat_experiment.get("prereg_commit"),
            "pr_number": flat_experiment.get("pr_number"),
            "tracking_issue": flat_experiment.get("tracking_issue"),
        })

        data["contract"] = {
            "identity": identity_block,
            "problem": flat_problem,
            "config": config_block,
            "runs": flat_runs,
            "servers": flat_servers,
            "custom_env": flat_custom_env,
            "max_generations": flat_experiment.get("max_generations", 25),
            "stopping_rule": stopping_rule,
            "baseline": flat_baseline,
            "tools": flat_tools,
        }

        # Cron IDs historically lived inside ``launch.*``. Move them to the
        # control plane where they belong in v2.
        launch_extras: dict[str, Any] = {k: v for k, v in flat_launch.items()}
        anomaly_cron = launch_extras.pop("anomaly_detector_cron_id", None)
        checkpoint_cron = launch_extras.pop("checkpoint_cron_id", None)
        watchdog_pid = launch_extras.pop("watchdog_pid", None)

        data["lifecycle"] = {
            "status": flat_experiment.get("status", "preregistered"),
            "launch": launch_extras,
            "smoke_test": flat_smoke,
            "treatment_verification": flat_tv,
        }

        # treatment_checks can arrive in multiple v1 shapes. Only accept the
        # v2 shape (dict with ``results`` list); otherwise drop to defaults.
        tc_block: dict[str, Any] = {}
        if (
            isinstance(flat_tc, dict)
            and "results" in flat_tc
            and isinstance(flat_tc.get("results"), list)
        ):
            tc_block = flat_tc

        data["telemetry"] = {
            "checkpoints": flat_checkpoints,
            "mid_run_test_eval": flat_mid_run,
            "checkpoint_analysis": flat_ca,
            "treatment_checks": tc_block,
        }

        data["control_plane"] = {
            "watchdog": flat_watchdog,
            "watchdog_pid": watchdog_pid,
            "anomaly_detector_cron_id": anomaly_cron,
            "checkpoint_cron_id": checkpoint_cron,
        }

        # Drop the flat keys so ``extra="allow"`` doesn't also carry them.
        for k in _LEGACY_FLAT_KEYS:
            data.pop(k, None)

        return data

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
            if not self.contract.servers:
                errors.append(
                    f"contract.servers[] must be non-empty for status={status}. "
                    f"Add the server hostnames used by this experiment."
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

    # ---- flat compatibility views ---------------------------------------
    # These properties exist so legacy readers (``m.runs``, ``m.experiment.status``,
    # ``m.watchdog.plugin``, ...) keep working while the codebase transitions
    # to nested-only access. New code should read ``self.contract.*`` /
    # ``self.lifecycle.*`` / ``self.telemetry.*`` / ``self.control_plane.*``
    # directly — these views will be removed in a follow-up cleanup.

    @property
    def experiment(self) -> ExperimentSection:
        """Legacy flat ``experiment:`` view derived from nested storage."""
        return ExperimentSection(
            name=self.contract.identity.name,
            task=self.contract.identity.task,
            status=self.lifecycle.status,
            branch=self.contract.identity.branch,
            max_generations=self.contract.max_generations,
            pr_number=self.contract.identity.pr_number,
            tracking_issue=self.contract.identity.tracking_issue,
            prereg_commit=self.contract.identity.prereg_commit,
            stopping_rule=self.contract.stopping_rule.description,
        )

    @property
    def runs(self) -> list[RunSpec]:
        """Legacy flat ``runs`` view — use ``self.contract.runs`` in new code."""
        return self.contract.runs

    @property
    def servers(self) -> list[str]:
        """Legacy flat ``servers`` view — use ``self.contract.servers``."""
        return self.contract.servers

    @property
    def problem(self) -> ProblemSpec:
        """Legacy flat ``problem`` view — use ``self.contract.problem``."""
        return self.contract.problem

    @property
    def config(self) -> dict[str, Any]:
        """Legacy flat ``config`` dict view — merge of typed + extra fields."""
        cs = self.contract.config
        merged: dict[str, Any] = dict(cs.extra or {})
        for k in (
            "problem_name",
            "pipeline",
            "prompt_fetcher",
            "evolution",
            "llm_model",
            "n_workers",
            "max_generations",
        ):
            v = getattr(cs, k, None)
            if v is not None:
                merged[k] = v
        return merged

    @property
    def custom_env(self) -> dict[str, str]:
        """Legacy flat ``custom_env`` view."""
        return self.contract.custom_env

    @property
    def baseline(self) -> BaselineInfo:
        """Legacy flat ``baseline`` view."""
        return self.contract.baseline

    @property
    def tools(self) -> list[ToolRef]:
        """Legacy flat ``tools`` view."""
        return self.contract.tools

    @property
    def launch(self) -> LaunchInfo:
        """Legacy flat ``launch`` view — use ``self.lifecycle.launch``."""
        return self.lifecycle.launch

    @property
    def smoke_test(self) -> SmokeTestInfo:
        """Legacy flat ``smoke_test`` view."""
        return self.lifecycle.smoke_test

    @property
    def treatment_verification(self) -> TreatmentVerificationInfo:
        """Legacy flat ``treatment_verification`` view."""
        return self.lifecycle.treatment_verification

    @property
    def watchdog(self) -> WatchdogSection:
        """Legacy flat ``watchdog`` view — use ``self.control_plane.watchdog``."""
        return self.control_plane.watchdog

    @property
    def checkpoints(self) -> list[CheckpointEntry]:
        """Legacy flat ``checkpoints`` view — use ``self.telemetry.checkpoints``."""
        return self.telemetry.checkpoints

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

        # Read current status from nested v2 shape, falling back to legacy flat.
        current = (
            (raw.get("lifecycle") or {}).get("status")
            or (raw.get("experiment") or {}).get("status")
            or "preregistered"
        )

        allowed = VALID_TRANSITIONS.get(current, set())
        if allow_recovery:
            allowed = allowed | RECOVERY_TRANSITIONS.get(current, set())

        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition: {current} -> {new_status}. Allowed: {allowed}"
            )

        # Write to the canonical nested path. If a legacy flat ``experiment:``
        # block still exists (pre-flatten yamls), keep it in sync so other
        # readers that haven't migrated yet still see a consistent value.
        raw.setdefault("lifecycle", {})["status"] = new_status
        if isinstance(raw.get("experiment"), dict):
            raw["experiment"]["status"] = new_status

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
