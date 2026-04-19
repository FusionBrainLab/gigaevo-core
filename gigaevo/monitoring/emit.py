"""Canonical-event emission helper.

Single point of truth for converting a validated `BaseEvent` into a structured
loguru log line in the `[EVENT_NAME] {json}` format that `log_audit.py`
already parses.

Records carry `extra["canonical_event"] = True` so sinks that emit follow-up
events (notably the EXCEPTION sink) can skip canonical-event lines and avoid
infinite recursion.

Track B4 — Redis minute-bucket counters
---------------------------------------
When `configure_event_counters(client, prefix)` is called once at process
start, every `emit()` additionally INCRs ``{prefix}:events:{event}:{minute}``
with a week-long TTL. The INCR is silent: no loguru line, failures swallowed,
and `emit()` never raises on Redis trouble. The watchdog reads these counters
to fire the generic `EVENT_RATE_ZERO` alert when a registered event with
``expected_after_gen > 0`` has produced zero emissions past its threshold.
"""

from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger

from gigaevo.monitoring.events import BaseEvent

# Process-scoped counter configuration. Populated by
# ``configure_event_counters`` at run startup; cleared by
# ``reset_event_counters`` (used by tests).
_event_redis: Any | None = None
_event_prefix: str = ""

# TTL for bucket keys. A week is enough for retrospective auditing while
# keeping Redis memory bounded.
_COUNTER_TTL_SECONDS = 7 * 24 * 3600


def configure_event_counters(*, client: Any, prefix: str) -> None:
    """Enable Redis INCR counters for every subsequent ``emit()``.

    Called once per run process (see ``run.py``). Passing an empty prefix
    leaves counters disabled — there is nowhere well-namespaced to write.
    """
    global _event_redis, _event_prefix
    if not prefix:
        # No prefix → nowhere to write. Leave counters disabled.
        _event_redis = None
        _event_prefix = ""
        return
    _event_redis = client
    _event_prefix = prefix


def reset_event_counters() -> None:
    """Clear process-level counter state. Intended for tests."""
    global _event_redis, _event_prefix
    _event_redis = None
    _event_prefix = ""


def configure_event_counters_from_cfg(cfg: Any) -> None:
    """One-call wiring from a Hydra-resolved config.

    Keeps ``run.py`` a thin entrypoint: the process owns the sync Redis
    client used for minute-bucket INCRs here (separate from the async
    program-storage client), and we never leak its construction into the
    caller. Silent no-op when the prefix is absent.
    """
    prefix = cfg.redis_storage.config.key_prefix
    if not prefix:
        return
    import redis

    client = redis.Redis(
        host=cfg.redis.host,
        port=cfg.redis.port,
        db=cfg.redis.db,
    )
    configure_event_counters(client=client, prefix=prefix)


def _incr_counter(event_name: str) -> None:
    """INCR the minute bucket for this event. Silent on any error.

    Event emission must not block on Redis availability; if the client is
    absent, prefix is unset, or the call raises, we swallow and return.
    """
    client = _event_redis
    prefix = _event_prefix
    if client is None or not prefix or not event_name:
        return
    minute = int(time.time() // 60)
    key = f"{prefix}:events:{event_name}:{minute}"
    try:
        client.incr(key)
        client.expire(key, _COUNTER_TTL_SECONDS)
    except Exception:
        # Counter is diagnostic, not load-bearing. Never break emit().
        return


def emit(event: BaseEvent) -> None:
    """Emit a canonical event as a single structured loguru line.

    The caller is expected to construct the Pydantic event (which validates
    at construction time). This function only formats and logs it. The event
    name (ClassVar, not a field) is injected into the JSON body so auditor
    parsers see a self-describing record.
    """
    if not isinstance(event, BaseEvent):
        raise TypeError(
            f"emit() expects a BaseEvent instance, got {type(event).__name__}"
        )
    name = type(event).event
    if not name:
        raise ValueError(
            "emit() refuses to log an event with an empty event name — "
            "did you instantiate the abstract BaseEvent directly?"
        )
    payload = event.model_dump(mode="json")
    payload = {"event": name, **payload}
    logger.bind(canonical_event=True, event_name=name).info(
        f"[{name}] {json.dumps(payload, ensure_ascii=False)}"
    )
    _incr_counter(name)
