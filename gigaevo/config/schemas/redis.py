from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from gigaevo.config.schemas._base import FrozenStrictModel

if TYPE_CHECKING:
    from gigaevo.database.redis.config import RedisProgramStorageConfig


class RedisConfig(FrozenStrictModel):
    """Connection coordinates that resolve to a single Redis logical DB.

    The runtime construction site (``RedisProgramStorage``) accepts a
    pre-formed ``redis_url``; this schema exposes the host/port/db
    tuple and derives the URL via a computed field.
    """

    host: str = Field(
        default="localhost",
        min_length=1,
        description="Redis server hostname or IP address.",
    )
    port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis server TCP port.",
    )
    db: int = Field(
        default=0,
        ge=0,
        description="Redis logical database index for this run's program storage.",
    )
    resume: bool = Field(
        default=False,
        description="When true, reuse existing keys under the prefix instead of clearing them at startup.",
    )

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"

    def to_storage_config(
        self,
        *,
        key_prefix: str,
        max_connections: int = 100,
        connection_pool_timeout: float = 60.0,
        health_check_interval: int = 180,
        max_retries: int = 5,
        retry_delay: float = 0.2,
    ) -> RedisProgramStorageConfig:
        from gigaevo.database.redis.config import RedisProgramStorageConfig

        return RedisProgramStorageConfig(
            redis_url=self.url,
            key_prefix=key_prefix,
            max_connections=max_connections,
            connection_pool_timeout=connection_pool_timeout,
            health_check_interval=health_check_interval,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )


class DataPlaneSettings(FrozenStrictModel):
    """Tunables for the DataPlane coordinator.

    ``RedisConfig`` is referenced by composition so the dataplane and the
    program storage share a single Redis backing without a second URL
    declaration in the experiment file.
    """

    redis: RedisConfig
    key_prefix: str = Field(
        min_length=1,
        description="Redis key prefix that namespaces this experiment; must equal gigaevo:{experiment.name}.",
    )
    max_connections: int = Field(
        default=16,
        ge=1,
        description="Upper bound on the connection pool used by the dataplane coordinator.",
    )
    startup_timeout_s: float = Field(
        default=10.0,
        gt=0.0,
        description="Seconds to wait for the dataplane to become ready before failing startup.",
    )
