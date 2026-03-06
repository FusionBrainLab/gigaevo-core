from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import AnyUrl, BaseModel, Field


class RedisConnectionConfig(BaseModel):
    """Configuration for Redis connection."""

    redis_url: AnyUrl = Field(default="redis://localhost:6379/0")
    max_connections: int = Field(default=100, ge=10)
    connection_pool_timeout: float = Field(default=60.0, ge=1.0)
    health_check_interval: int = Field(default=180, ge=1)
    max_retries: int = Field(default=5, ge=1)
    retry_delay: float = Field(default=0.2, ge=0.0)

    model_config = {"extra": "forbid"}


class RedisKeyConfig(BaseModel):
    """Configuration for Redis key templates."""

    key_prefix: str = Field(default="gigaevo")
    program_key_tpl: str = Field(default="{prefix}:program:{pid}")
    status_stream_tpl: str = Field(default="{prefix}:status_events")
    status_set_tpl: str = Field(default="{prefix}:status:{status}")

    model_config = {"extra": "forbid"}


class RedisLockConfig(BaseModel):
    """Configuration for instance locking."""

    lock_expiry_secs: int = Field(
        default=300, description="Lock TTL in seconds (5 min)"
    )
    lock_renewal_secs: int = Field(
        default=120, description="Lock renewal interval in seconds (2 min)"
    )

    model_config = {"extra": "forbid"}


class RedisProgramStorageConfig(BaseModel):
    """Configuration for Redis program storage."""

    # Connection settings
    redis_url: AnyUrl = Field(default="redis://localhost:6379/0")
    max_connections: int = Field(default=100, ge=10)
    connection_pool_timeout: float = Field(default=60.0, ge=1.0)
    health_check_interval: int = Field(default=180, ge=1)
    max_retries: int = Field(default=5, ge=1)
    retry_delay: float = Field(default=0.2, ge=0.0)

    # Key settings
    key_prefix: str = Field(default="gigaevo")
    program_key_tpl: str = Field(default="{prefix}:program:{pid}")
    status_stream_tpl: str = Field(default="{prefix}:status_events")
    status_set_tpl: str = Field(default="{prefix}:status:{status}")

    # Behavior
    merge_strategy: str | Callable[..., Any] = Field(default="additive")
    metrics_interval: float = Field(
        default=1.0, ge=0.1, description="Interval (s) for Redis metrics collection"
    )
    read_only: bool = Field(
        default=False,
        description="If True, skip instance locking and disable write operations.",
    )

    # Locking
    lock_expiry_secs: int = Field(default=300)
    lock_renewal_secs: int = Field(default=120)

    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    def to_connection_config(self) -> RedisConnectionConfig:
        return RedisConnectionConfig(
            redis_url=self.redis_url,
            max_connections=self.max_connections,
            connection_pool_timeout=self.connection_pool_timeout,
            health_check_interval=self.health_check_interval,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

    def to_key_config(self) -> RedisKeyConfig:
        return RedisKeyConfig(
            key_prefix=self.key_prefix,
            program_key_tpl=self.program_key_tpl,
            status_stream_tpl=self.status_stream_tpl,
            status_set_tpl=self.status_set_tpl,
        )

    def to_lock_config(self) -> RedisLockConfig:
        return RedisLockConfig(
            lock_expiry_secs=self.lock_expiry_secs,
            lock_renewal_secs=self.lock_renewal_secs,
        )
