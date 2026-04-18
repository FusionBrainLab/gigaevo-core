from __future__ import annotations

from collections.abc import Iterable
import json
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import redis as redis_lib

from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot

_log = logger.bind(component="redis_queries")


def _coerce_count(raw: Any) -> int:
    """Best-effort coercion of a Redis INCR return (bytes/str/int) to int."""
    if raw is None:
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, bytes):
        try:
            return int(raw.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            return 0
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return 0
    return 0


def count_events_in_window(
    r: redis_lib.Redis,
    prefix: str,
    event_name: str,
    window_minutes: int = 10,
    now_minute: int | None = None,
) -> int:
    """Sum the last ``window_minutes`` of the minute-bucket counter.

    Keys follow the ``{prefix}:events:{event_name}:{minute}`` schema written
    by ``gigaevo.monitoring.emit``. Returns 0 on any Redis error — the
    watchdog must never fail because the auxiliary counter table is down.
    """
    if window_minutes <= 0:
        return 0
    now = now_minute if now_minute is not None else int(time.time() // 60)
    first = now - (window_minutes - 1)
    keys = [f"{prefix}:events:{event_name}:{m}" for m in range(first, now + 1)]
    try:
        values = r.mget(keys)
    except Exception:
        return 0
    return sum(_coerce_count(v) for v in values)


def collect_event_window_counts(
    r: redis_lib.Redis,
    prefix: str,
    event_names: Iterable[str],
    window_minutes: int = 10,
    now_minute: int | None = None,
) -> dict[str, int]:
    """Batch-read minute-bucket counts for multiple events in one MGET.

    Given N events and W buckets, assembles all N*W keys in order and issues
    a single ``MGET``, then slices the result per event. One round-trip
    regardless of how many events are registered in ``CANONICAL_EVENTS``.
    """
    names = list(event_names)
    if not names or window_minutes <= 0:
        return {n: 0 for n in names}
    now = now_minute if now_minute is not None else int(time.time() // 60)
    first = now - (window_minutes - 1)
    keys: list[str] = []
    for name in names:
        keys.extend(f"{prefix}:events:{name}:{m}" for m in range(first, now + 1))
    try:
        values = r.mget(keys)
    except Exception:
        return {n: 0 for n in names}
    out: dict[str, int] = {}
    for i, name in enumerate(names):
        slab = values[i * window_minutes : (i + 1) * window_minutes]
        out[name] = sum(_coerce_count(v) for v in slab)
    return out


def get_generation(r: redis_lib.Redis, prefix: str) -> int | None:
    """Get the current generation count from run_state hash.

    This is the CANONICAL source of generation count.
    Never use log grep or metric step values.
    """
    raw = r.hget(f"{prefix}:run_state", "engine:total_generations")
    if raw is None:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def get_frontier_metrics(
    r: redis_lib.Redis, prefix: str, metric_names: list[str]
) -> dict[str, float | None]:
    """Get latest frontier value for each metric.

    Reads the last entry from each metric's history list.
    """
    result: dict[str, float | None] = {}
    for name in metric_names:
        key = f"{prefix}:metrics:history:program_metrics:valid_frontier_{name}"
        raw = r.lindex(key, -1)
        if raw is None:
            result[name] = None
            continue
        try:
            result[name] = json.loads(raw)["v"]
        except (KeyError, json.JSONDecodeError, TypeError, ValueError):
            _log.warning(f"Malformed frontier entry for {prefix}/{name}: {raw!r}")
            result[name] = None
    return result


def get_program_counts(
    r: redis_lib.Redis, prefix: str
) -> tuple[int | None, int | None]:
    """Get (total_programs, valid_programs) from metrics history."""
    total = None
    valid = None
    raw_total = r.lindex(
        f"{prefix}:metrics:history:program_metrics:programs_total_count", -1
    )
    raw_valid = r.lindex(
        f"{prefix}:metrics:history:program_metrics:programs_valid_count", -1
    )
    if raw_total is not None:
        try:
            total = int(json.loads(raw_total)["v"])
        except (KeyError, json.JSONDecodeError, TypeError, ValueError):
            pass
    if raw_valid is not None:
        try:
            valid = int(json.loads(raw_valid)["v"])
        except (KeyError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return total, valid


def get_validator_duration(
    r: redis_lib.Redis, prefix: str
) -> tuple[float | None, float | None]:
    """Get (mean, max) validator stage duration from last 20 entries.

    Uses bounded LRANGE -20 -1 to avoid reading unbounded lists.
    """
    key = (
        f"{prefix}:metrics:history:dag_runner:dag:internals:"
        "CallValidatorFunction:stage_duration"
    )
    # Check key type first to avoid errors on wrong type
    key_type = r.type(key)
    if key_type not in ("list", b"list"):
        return None, None

    recent = r.lrange(key, -20, -1)
    durations: list[float] = []
    for raw in recent:
        try:
            v = json.loads(raw)["v"]
            if v is not None:
                durations.append(float(v))
        except (KeyError, json.JSONDecodeError, TypeError, ValueError):
            continue

    if not durations:
        return None, None
    return sum(durations) / len(durations), max(durations)


def get_status_counts(r: redis_lib.Redis, prefix: str) -> dict[str, int]:
    """Get program status set cardinalities.

    Returns counts for DONE, QUEUED, RUNNING, DISCARDED.
    """
    return {
        "DONE": r.scard(f"{prefix}:status:DONE"),
        "QUEUED": r.scard(f"{prefix}:status:QUEUED"),
        "RUNNING": r.scard(f"{prefix}:status:RUNNING"),
        "DISCARDED": r.scard(f"{prefix}:status:DISCARDED"),
    }


def collect_snapshot(
    r: redis_lib.Redis,
    run_spec: RunSpec,
    metric_names: list[str] | None = None,
    pid: int | None = None,
    event_window_minutes: int = 10,
) -> RunSnapshot:
    """Collect a complete RunSnapshot from Redis for one run.

    This is the primary composition function -- it calls all individual
    query functions and assembles the result into a RunSnapshot.

    Never writes to Redis. All operations are read-only.

    Args:
        r: Redis client connected to the correct DB for this run.
        run_spec: Parsed run specification.
        metric_names: Metric names to query. Defaults to ["fitness"].
        pid: Optional PID to check liveness for.
        event_window_minutes: Size of the minute-bucket window used to
            sum canonical-event counts. Set to 0 to skip the batch MGET
            entirely — useful for tests or Redis-less contexts.

    Returns:
        RunSnapshot with all available data. On Redis errors,
        returns RunSnapshot.empty() with the error message.
    """
    if metric_names is None:
        metric_names = ["fitness"]

    try:
        gen = get_generation(r, run_spec.prefix)
        metrics = get_frontier_metrics(r, run_spec.prefix, metric_names)
        total, valid = get_program_counts(r, run_spec.prefix)
        val_mean, val_max = get_validator_duration(r, run_spec.prefix)
        status_counts = get_status_counts(r, run_spec.prefix)
        total_keys = r.dbsize()
        completion_reason = r.hget(
            f"{run_spec.prefix}:run_state", "engine:completion_reason"
        )

        # Track B4: single MGET across all registered canonical events.
        # ``None`` when disabled (window=0) or when the registry is empty,
        # so AlertDetector's EVENT_RATE_ZERO predicate stays silent in
        # those cases (matches backward-compat semantics).
        event_window_counts: dict[str, int] | None = None
        if event_window_minutes > 0:
            from gigaevo.monitoring.events import CANONICAL_EVENTS

            if CANONICAL_EVENTS:
                event_window_counts = collect_event_window_counts(
                    r,
                    run_spec.prefix,
                    list(CANONICAL_EVENTS.keys()),
                    window_minutes=event_window_minutes,
                )

        pid_alive = None
        if pid is not None:
            import os

            try:
                os.kill(pid, 0)
                pid_alive = True
            except (ProcessLookupError, PermissionError):
                pid_alive = False

        return RunSnapshot(
            run_spec=run_spec,
            generation=gen,
            metrics=metrics,
            total_programs=total,
            valid_programs=valid,
            running_programs=status_counts.get("RUNNING"),
            queued_programs=status_counts.get("QUEUED"),
            done_programs=status_counts.get("DONE"),
            validator_mean_s=val_mean,
            validator_max_s=val_max,
            total_keys=total_keys,
            pid=pid,
            pid_alive=pid_alive,
            completed=completion_reason is not None,
            completion_reason=completion_reason,
            error=None,
            event_window_counts=event_window_counts,
        )
    except Exception as exc:
        _log.error(f"Failed to collect snapshot for {run_spec}: {exc}")
        return RunSnapshot(
            run_spec=run_spec,
            error=str(exc),
        )
