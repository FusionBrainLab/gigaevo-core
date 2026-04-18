"""Canonical-event emission helper.

Single point of truth for converting a validated `BaseEvent` into a structured
loguru log line in the `[EVENT_NAME] {json}` format that `log_audit.py`
already parses.

Records carry `extra["canonical_event"] = True` so sinks that emit follow-up
events (notably the EXCEPTION sink) can skip canonical-event lines and avoid
infinite recursion.
"""

from __future__ import annotations

import json

from loguru import logger

from gigaevo.monitoring.events import BaseEvent


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
