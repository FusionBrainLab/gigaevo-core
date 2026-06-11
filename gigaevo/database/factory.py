"""Single non-Hydra construction point for ProgramStorage backends.

Production engines get storage from Hydra (config/redis/default.yaml).
CLI tools and offline analytics that resolve runs dynamically
(prefix@db) construct read-only instances HERE — nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)


@dataclass
class RedisRunConfig:
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_prefix: str = ""
    label: str = ""

    def url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    def display_label(self) -> str:
        return self.label or f"{self.redis_prefix}@{self.redis_db}"


def build_readonly_redis_storage(
    *,
    host: str,
    port: int,
    db: int,
    key_prefix: str,
    max_connections: int = 50,
    connection_pool_timeout: float = 30.0,
    health_check_interval: int = 60,
) -> ProgramStorage:
    """Read-only storage for analytics/CLI paths. Never use for a live engine."""
    return RedisProgramStorage(
        RedisProgramStorageConfig(
            redis_url=f"redis://{host}:{port}/{db}",  # type: ignore[arg-type]  # pydantic validates str -> AnyUrl
            key_prefix=key_prefix,
            max_connections=max_connections,
            connection_pool_timeout=connection_pool_timeout,
            health_check_interval=health_check_interval,
            read_only=True,
        )
    )


def build_writable_redis_storage(
    *,
    host: str,
    port: int,
    db: int,
    key_prefix: str,
    max_connections: int = 50,
    connection_pool_timeout: float = 30.0,
    health_check_interval: int = 60,
) -> ProgramStorage:
    """Writable storage for offline tools (profiler, benchmarks). Not for live engines.

    Production engines always get storage from Hydra (config/redis/default.yaml).
    """
    return RedisProgramStorage(
        RedisProgramStorageConfig(
            redis_url=f"redis://{host}:{port}/{db}",  # type: ignore[arg-type]  # pydantic validates str -> AnyUrl
            key_prefix=key_prefix,
            max_connections=max_connections,
            connection_pool_timeout=connection_pool_timeout,
            health_check_interval=health_check_interval,
            read_only=False,
        )
    )
