"""Redis connection + manifest locking + atomic file writes.

Low-level primitives for concurrency-safe manifest mutation:

- :func:`get_redis` — connect to Redis (db 0) with actionable errors
- :func:`acquire_lock` / :func:`release_lock` — Redis-based mutual exclusion
- :func:`write_manifest_atomic` — write YAML via tmp + rename (FUSE-safe)

Public API used by :mod:`gigaevo.experiment.manifest` and tests.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import redis
import yaml


def get_redis() -> redis.Redis:
    """Get a Redis connection for manifest locking.

    Configuration:
        REDIS_HOST (env): Redis hostname (default: "localhost")
        REDIS_PORT (env): Redis port (default: 6379)

    Uses database 0 for locking and DB claims. This follows the same pattern
    as gigaevo.utils.redis.RedisRunConfig for consistency.

    Returns:
        A Redis client configured with synchronous operations.

    Raises:
        RuntimeError: If Redis connection fails (unavailable, port error, etc).
            Solution: Ensure Redis is running with: redis-server or check REDIS_HOST/REDIS_PORT.
    """
    host = os.environ.get("REDIS_HOST", "localhost")
    port_str = os.environ.get("REDIS_PORT", "6379")

    try:
        port = int(port_str)
    except ValueError:
        raise RuntimeError(
            f"Invalid REDIS_PORT='{port_str}' (must be an integer). "
            f"Fix: export REDIS_PORT=6379 or check environment variables."
        ) from None

    try:
        r = redis.Redis(host=host, port=port, db=0)
        r.ping()
        return r
    except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
        raise RuntimeError(
            f"Cannot connect to Redis at {host}:{port}. "
            f"Fix: Start Redis with `redis-server` or set REDIS_HOST/REDIS_PORT. "
            f"Error: {e}"
        ) from e


def acquire_lock(r: redis.Redis, experiment: str, timeout: float = 5.0) -> str:
    """Acquire Redis-based lock. Returns lock key. Raises on timeout."""
    lock_key = f"experiments:{experiment}:yaml_lock"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if r.set(lock_key, str(os.getpid()), nx=True, ex=30):
            return lock_key
        time.sleep(0.25)
    raise RuntimeError(
        f"Could not acquire lock {lock_key} after {timeout}s. "
        f"Current holder: {r.get(lock_key)}"
    )


def release_lock(r: redis.Redis, lock_key: str) -> None:
    r.delete(lock_key)


def write_manifest_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write YAML atomically via tmp + rename."""
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


__all__ = [
    "get_redis",
    "acquire_lock",
    "release_lock",
    "write_manifest_atomic",
]
