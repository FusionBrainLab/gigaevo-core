"""Shared utilities for adversarial co-evolution problems.

Provides opponent archive reading and code execution, following the patterns
established in gigaevo/prompts/fetcher.py (archive reading) and
gigaevo/prompts/coevolution/stages.py (code execution via exec/compile).

Used by adversarial validate.py files to read and execute opponent programs.
"""

from __future__ import annotations

import json
import logging
import random
import signal
import time
from typing import Any

import redis

logger = logging.getLogger(__name__)

# --- Module-level cache (same pattern as GigaEvoArchivePromptFetcher) ---
_cache: list[tuple[str, float, str]] | None = None
_cache_ts: float = 0.0
_CACHE_TTL: float = 30.0


def read_opponent_archive(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    prefix: str = "",
    fitness_key: str = "fitness",
    cache_ttl: float = _CACHE_TTL,
) -> list[tuple[str, float, str]]:
    """Read all programs from an opponent's MAP-Elites archive.

    Follows the exact pattern of GigaEvoArchivePromptFetcher._refresh_candidates()
    from gigaevo/prompts/fetcher.py:251-299.

    Returns list of (program_id, fitness, code) sorted by fitness descending.
    """
    global _cache, _cache_ts

    now = time.monotonic()
    if _cache is not None and (now - _cache_ts) < cache_ttl:
        return _cache

    try:
        r = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        archive_key = f"{prefix}:archive"
        program_ids = list(r.hvals(archive_key))
        if not program_ids:
            _cache = []
            _cache_ts = now
            return _cache

        candidates: list[tuple[str, float, str]] = []
        for pid in program_ids:
            program_key = f"{prefix}:program:{pid}"
            raw = r.get(program_key)
            if not raw:
                continue
            try:
                data = json.loads(raw)
                metrics = data.get("metrics", {})
                fitness = float(metrics.get(fitness_key, 0.0))
                code = data.get("code", "")
                if code:
                    candidates.append((pid, fitness, code))
            except Exception as exc:
                logger.debug(
                    "[adversarial.shared] Error parsing program %s: %s", pid, exc
                )

        candidates.sort(key=lambda x: x[1], reverse=True)
        _cache = candidates
        _cache_ts = now
        return _cache

    except Exception as exc:
        logger.warning("[adversarial.shared] Redis read error: %s", exc)
        return _cache if _cache is not None else []


def sample_opponents(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    prefix: str = "",
    fitness_key: str = "fitness",
    n: int = 5,
) -> list[tuple[str, float, str]]:
    """Sample n opponents with fitness-proportional selection."""
    archive = read_opponent_archive(
        host=host, port=port, db=db, prefix=prefix, fitness_key=fitness_key
    )
    if not archive:
        return []
    n = min(n, len(archive))
    weights = [f + 1e-6 for _, f, _ in archive]
    return random.choices(archive, weights=weights, k=n)


def exec_entrypoint(code: str, timeout: float = 5.0) -> Any:
    """Execute opponent code and return entrypoint() result.

    Follows the pattern from PromptExecutionStage.compute()
    (gigaevo/prompts/coevolution/stages.py:63-76) with added signal-based
    timeout protection for individual opponent calls within the validator.

    Args:
        code: Python source code containing a def entrypoint() function.
        timeout: Maximum execution time in seconds (SIGALRM-based).

    Returns:
        The return value of entrypoint().

    Raises:
        TimeoutError: If execution exceeds timeout.
        ValueError: If code has no entrypoint or fails to execute.
    """

    def _handler(signum, frame):
        raise TimeoutError(f"Opponent execution exceeded {timeout}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        namespace: dict[str, Any] = {}
        exec(compile(code, "<opponent_program>", "exec"), namespace)  # noqa: S102
        entrypoint_fn = namespace.get("entrypoint")
        if not callable(entrypoint_fn):
            raise ValueError("Opponent program has no callable entrypoint()")
        return entrypoint_fn()
    except (TimeoutError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"Opponent entrypoint() failed: {exc}") from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def get_opponent_config() -> dict[str, Any]:
    """Read opponent connection details from Hydra overrides (env vars).

    Set by launch.sh via extra_overrides or env_updates in experiment.yaml:
        OPPONENT_REDIS_HOST, OPPONENT_REDIS_PORT, OPPONENT_REDIS_DB, OPPONENT_PREFIX

    Returns dict with keys: host, port, db, prefix. Empty prefix means no opponent.
    """
    import os

    return {
        "host": os.environ.get("OPPONENT_REDIS_HOST", "localhost"),
        "port": int(os.environ.get("OPPONENT_REDIS_PORT", "6379")),
        "db": int(os.environ.get("OPPONENT_REDIS_DB", "0")),
        "prefix": os.environ.get("OPPONENT_PREFIX", ""),
    }


def reset_cache() -> None:
    """Reset module-level cache (for testing)."""
    global _cache, _cache_ts
    _cache = None
    _cache_ts = 0.0
