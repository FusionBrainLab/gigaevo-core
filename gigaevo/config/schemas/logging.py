from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, field_validator

from gigaevo.config.schemas._base import FrozenStrictModel, reject_empty_or_cwd_path

if TYPE_CHECKING:
    from gigaevo.utils.trackers.core import GenericLogger


_LOG_LEVELS = Literal[
    "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
]


class LoggingSettings(FrozenStrictModel):
    """File-logger knobs that wrap loguru's add() parameters.

    The ``log_dir`` field is None when the experiment defers to the
    runtime CLI to write logs under ``output_dir/experiment_id/``;
    setting it explicitly pins logs to a fixed location regardless of
    output_dir."""

    level: _LOG_LEVELS = Field(
        default="INFO",
        description="Minimum loguru log level captured to file.",
    )
    log_dir: Path | None = Field(
        default=None,
        description="Directory for log files; None defers to output_dir/experiment_id/.",
    )
    rotation: str = Field(
        default="50 MB",
        min_length=1,
        description="Loguru rotation specifier; size or time string accepted by loguru.add.",
    )
    retention: str = Field(
        default="30 days",
        min_length=1,
        description="Loguru retention specifier controlling how long rotated logs are kept.",
    )

    @field_validator("log_dir")
    @classmethod
    def _log_dir_not_empty(cls, value: Path | None) -> Path | None:
        return reject_empty_or_cwd_path("log_dir", value)


class TBTrackerConfig(FrozenStrictModel):
    """TensorBoard tracker. ``logdir`` is the directory the
    SummaryWriter writes events into; commonly nested under the
    experiment's ``output_dir`` and named ``tb_logs``."""

    kind: Literal["tensorboard"] = "tensorboard"
    logdir: Path = Field(
        description="Directory the TensorBoard SummaryWriter writes event files into.",
    )
    queue_size: int = Field(
        default=8192,
        ge=1,
        description="Async write queue capacity; the writer drops events when full.",
    )
    flush_secs: float = Field(
        default=3.0,
        gt=0.0,
        description="Background flush interval in seconds.",
    )

    def build(self) -> GenericLogger:
        from gigaevo.utils.trackers import init_tb
        from gigaevo.utils.trackers.configs import TBConfig

        return init_tb(
            TBConfig(logdir=self.logdir),
            queue_size=self.queue_size,
            flush_secs=self.flush_secs,
        )


class WandBTrackerConfig(FrozenStrictModel):
    """Weights & Biases tracker. ``project`` is required for non-local
    runs; ``name`` becomes the run label. ``entity`` and ``tags`` are
    optional."""

    kind: Literal["wandb"] = "wandb"
    project: str = Field(
        min_length=1,
        description="Weights & Biases project name.",
    )
    name: str | None = Field(
        default=None,
        description="Run display name shown in the W&B UI.",
    )
    entity: str | None = Field(
        default=None,
        description="W&B entity (team or username) owning the project.",
    )
    notes: str | None = Field(
        default=None,
        description="Free-form notes attached to the run.",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Tag list attached to the run for filtering.",
    )
    resume: bool = Field(
        default=False,
        description="When true, resume an existing W&B run instead of creating a new one.",
    )
    queue_size: int = Field(
        default=8192,
        ge=1,
        description="Async write queue capacity; the writer drops events when full.",
    )
    flush_secs: float = Field(
        default=3.0,
        gt=0.0,
        description="Background flush interval in seconds.",
    )

    def build(self) -> GenericLogger:
        from gigaevo.utils.trackers import init_wandb
        from gigaevo.utils.trackers.configs import WBConfig

        return init_wandb(
            WBConfig(
                project=self.project,
                name=self.name,
                entity=self.entity,
                notes=self.notes,
                tags=self.tags,
                resume=self.resume,
            ),
            queue_size=self.queue_size,
            flush_secs=self.flush_secs,
        )


class RedisMetricsTrackerConfig(FrozenStrictModel):
    """Redis-backed metrics tracker. Mirrors the runtime
    :class:`RedisMetricsConfig` surface — store_history toggles the
    time-series capture, max_history_per_metric caps the per-metric
    FIFO history."""

    kind: Literal["redis"] = "redis"
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        min_length=1,
        description="Full redis:// URL the tracker writes metrics to.",
    )
    key_prefix: str = Field(
        default="gigaevo:metrics",
        min_length=1,
        description="Redis key prefix under which metric streams are stored.",
    )
    store_history: bool = Field(
        default=True,
        description="When true the tracker retains a per-metric FIFO history.",
    )
    max_history_per_metric: int = Field(
        default=10_000,
        ge=1,
        description="Maximum samples retained per metric when history is enabled.",
    )
    max_connections: int = Field(
        default=10,
        ge=1,
        description="Upper bound on the tracker's Redis connection pool.",
    )
    socket_timeout: float = Field(
        default=5.0,
        ge=0.1,
        description="Per-operation socket timeout in seconds.",
    )
    queue_size: int = Field(
        default=8192,
        ge=1,
        description="Async write queue capacity; the writer drops events when full.",
    )
    flush_secs: float = Field(
        default=3.0,
        gt=0.0,
        description="Background flush interval in seconds.",
    )

    def build(self) -> GenericLogger:
        from gigaevo.utils.trackers import init_redis
        from gigaevo.utils.trackers.configs import RedisMetricsConfig

        return init_redis(
            RedisMetricsConfig(
                redis_url=self.redis_url,
                key_prefix=self.key_prefix,
                store_history=self.store_history,
                max_history_per_metric=self.max_history_per_metric,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
            ),
            queue_size=self.queue_size,
            flush_secs=self.flush_secs,
        )


TrackerConfig = Annotated[
    TBTrackerConfig | WandBTrackerConfig | RedisMetricsTrackerConfig,
    Field(discriminator="kind"),
]


class LoggingConfig(FrozenStrictModel):
    """Top-level logging composition. Holds the file-logger settings
    plus at least one metric tracker; ``build_writer`` fans them out
    through :class:`CompositeLogger`.

    The runtime :class:`CompositeLogger` raises on construction with
    zero backends — there is no observable distinction between
    "no metrics" and "metrics ignored", and the latter is more useful.
    Experiments that genuinely want zero metric output point at a
    Redis tracker bound to a local instance and ignore the writes;
    that matches production deployments where the Redis backend is
    the canonical write-and-discard path."""

    settings: LoggingSettings = Field(default_factory=lambda: LoggingSettings())
    trackers: list[TrackerConfig] = Field(
        min_length=1,
        description="One or more metric tracker backends; the writer fans events out to each.",
    )

    def build_writer(self) -> GenericLogger:
        from gigaevo.utils.trackers import init_composite

        return init_composite(*(t.build() for t in self.trackers))
