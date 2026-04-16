"""Redis connection + manifest locking + atomic file writes.

Low-level primitives for concurrency-safe manifest mutation:

- :func:`get_redis` — connect to Redis (db 0) with actionable errors
- :func:`acquire_lock` / :func:`release_lock` — Redis-based mutual exclusion
- :func:`read_manifest_rt` — read YAML preserving comments and key order
- :func:`write_manifest_atomic` — write YAML via tmp + rename (FUSE-safe)

Public API used by :mod:`gigaevo.experiment.manifest` and tests.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import time
from typing import Any

import redis
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


def _make_rt_yaml() -> YAML:
    """Build a configured round-trip YAML parser/emitter.

    Emits ``None`` as the explicit literal ``null`` so that diffs stay quiet
    when a tool reads → mutates → writes a manifest (ruamel's default would
    drop the value entirely, producing churn against the prior PyYAML output).
    """
    y = YAML(typ="rt")
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=2, offset=0)
    y.width = 4096
    y.representer.add_representer(
        type(None),
        lambda repr_, _: repr_.represent_scalar("tag:yaml.org,2002:null", "null"),
    )
    return y


_RT_YAML = _make_rt_yaml()


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


def read_manifest_rt(path: Path) -> CommentedMap:
    """Round-trip read of ``experiment.yaml`` via ruamel.yaml.

    Preserves comments, key order, and quoting so a subsequent
    :func:`write_manifest_atomic` call leaves untouched fields byte-stable.
    Always returns a ``CommentedMap`` (which is a regular ``MutableMapping``
    that Pydantic accepts).
    """
    with open(path) as f:
        data = _RT_YAML.load(f)
    if data is None:
        return CommentedMap()
    if not isinstance(data, CommentedMap):
        raise ValueError(f"{path} must be a YAML mapping, not {type(data).__name__}")
    return data


def write_manifest_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write YAML atomically via tmp + rename.

    Accepts either a ``CommentedMap`` (round-trip dump preserves comments
    and key order) or a plain ``dict`` (clean dump). The dump goes to a
    sibling ``.yaml.tmp`` first, then ``rename(2)`` to make the swap
    atomic on POSIX/FUSE filesystems.
    """
    buf = io.StringIO()
    _RT_YAML.dump(data, buf)
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


__all__ = [
    "get_redis",
    "acquire_lock",
    "release_lock",
    "read_manifest_rt",
    "write_manifest_atomic",
]
