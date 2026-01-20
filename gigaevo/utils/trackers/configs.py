from pathlib import Path
from typing import Any

from pydantic import AnyUrl, BaseModel, Field


class TBConfig(BaseModel):
    logdir: Path
    summary_writer_kwargs: dict[str, Any] = Field(default_factory=dict)


class WBConfig(BaseModel):
    project: str | None = None
    name: str | None = None
    entity: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    config: dict[str, Any] | None = None
    resume: bool = False


class RedisMetricsConfig(BaseModel):
    """Configuration for Redis metrics backend."""

    redis_url: AnyUrl = Field(default="redis://localhost:6379/0")
    key_prefix: str = Field(default="gigaevo:metrics")

    # Storage options
    store_history: bool = Field(
        default=True, description="Store time series history (not just latest values)"
    )
    max_history_per_metric: int = Field(
        default=10000, description="Max history entries per metric (FIFO)"
    )

    # Connection
    max_connections: int = Field(default=10, ge=1)
    socket_timeout: float = Field(default=5.0, ge=0.1)

    model_config = {"extra": "forbid"}
