"""Pydantic v2 schema for experiment.yaml.

Strict validation with actionable error messages, status-gated required fields,
and JSON Schema export. Lives alongside (does not replace) tools/experiment/manifest.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
import yaml

SUPPORTED_SCHEMA_VERSIONS = {1}
VALID_STATUSES = {"preregistered", "implemented", "running", "complete", "invalid"}


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
    alert_thresholds: AlertThresholds = AlertThresholds()
    poll_interval_s: int = 3600
    plot_retries: int = 3
    plot_retry_delay_s: int = 30
    rolling_comment_threshold_hours: int = 24
    checkpoint_milestones: list[float] = [0.1, 0.2, 0.5, 1.0]
    no_proxy_hosts: list[str] = []


class ManifestRunSpec(BaseModel):
    """One run within an experiment."""

    model_config = ConfigDict(extra="ignore")

    label: str
    db: int
    prefix: str
    pipeline: str
    problem_name: str
    condition: str
    chain_url: str | None = None
    mutation_url: str | None = None  # nullable in older experiments
    model_name: str
    pid: int | None = None
    log_path: str | None = None
    extra_overrides: list[str] | None = None
    run_env: dict[str, str] | None = None

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
    runs: list[ManifestRunSpec] = []
    servers: list[str] = []
    config: dict[str, Any] = {}
    custom_env: dict[str, str] = {}
    checkpoints: list[dict[str, Any]] = []
    launch: LaunchInfo = LaunchInfo()
    baseline: BaselineInfo = BaselineInfo()
    smoke_test: SmokeTestInfo = SmokeTestInfo()
    tools: list[dict[str, str]] = []
    watchdog: WatchdogSection = WatchdogSection()
    watchdog_plugin: str | None = None
    watchdog_plugin_options: dict[str, Any] | None = None

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
    def migrate_legacy_watchdog_fields(self) -> ExperimentManifest:
        """Migrate deprecated watchdog_plugin / watchdog_plugin_options to watchdog section."""
        watchdog_is_default = (
            self.watchdog.plugin is None and not self.watchdog.plot_metrics
        )
        if watchdog_is_default:
            if self.watchdog_plugin is not None:
                self.watchdog.plugin = self.watchdog_plugin
            if self.watchdog_plugin_options:
                plot_metrics = self.watchdog_plugin_options.get("plot_metrics", [])
                if plot_metrics:
                    self.watchdog.plot_metrics = list(plot_metrics)
        return self

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
    """Export the ExperimentManifest JSON Schema to a file.

    Use for editor autocompletion (VS Code, PyCharm) and CI validation.
    """
    schema = ExperimentManifest.model_json_schema()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")
