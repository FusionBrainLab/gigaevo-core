"""Tests for the canonical-event emission helper.

`emit(event: BaseEvent)` is the single point of truth for converting a
validated Pydantic event into a structured loguru log line in the
`[EVENT_NAME] {json}` format the existing auditor already parses.

Events are tagged with `extra["canonical_event"] = True` so sinks (e.g.
the EXCEPTION sink added in Phase 2.4) can distinguish canonical event
logs from the exceptions they emit on — preventing infinite recursion.
"""

from __future__ import annotations

import json
import re
from typing import ClassVar

from loguru import logger
import pytest

from gigaevo.monitoring.emit import emit
from gigaevo.monitoring.events import BaseEvent


class _PingEvent(BaseEvent):
    """Minimal concrete event for testing emit()."""

    event: ClassVar[str] = "__PING__"
    description: ClassVar[str] = "emit helper test"
    health_question: ClassVar[str] = "?"

    note: str


@pytest.fixture
def log_sink():
    """Capture loguru messages as a list of (message, record) tuples."""
    captured: list[tuple[str, dict]] = []

    def sink(message):
        record = message.record
        captured.append((message, dict(record)))

    sink_id = logger.add(sink, level="DEBUG", format="{message}")
    yield captured
    logger.remove(sink_id)


class TestEmitFormats:
    def test_emit_writes_single_log_line_with_event_tag_and_json_body(
        self, log_sink
    ) -> None:
        emit(_PingEvent(note="hi", run_label="K5_1_G"))

        messages = [str(m).strip() for m, _ in log_sink]
        # Exactly one [__PING__] line landed.
        ping_lines = [m for m in messages if "[__PING__]" in m]
        assert len(ping_lines) == 1, (
            f"expected exactly one [__PING__] line, got {ping_lines}"
        )
        line = ping_lines[0]
        # Body is valid JSON with the event fields.
        match = re.search(r"\[__PING__\]\s+(\{.*\})\s*$", line)
        assert match, f"line does not match [EVENT] {{json}} shape: {line!r}"
        body = json.loads(match.group(1))
        assert body["event"] == "__PING__"
        assert body["note"] == "hi"
        assert body["run_label"] == "K5_1_G"

    def test_emit_tags_record_with_canonical_event_marker(self, log_sink) -> None:
        emit(_PingEvent(note="tagged"))

        # Find the ping record.
        ping_records = [rec for m, rec in log_sink if "[__PING__]" in str(m)]
        assert len(ping_records) == 1
        rec = ping_records[0]
        # The record's `extra` dict must carry the canonical_event marker,
        # so downstream sinks can skip these lines (preventing recursion
        # when EXCEPTION sink re-logs on logger.exception).
        assert rec["extra"].get("canonical_event") is True


class TestEmitValidatesAtConstruction:
    def test_emit_accepts_any_BaseEvent_subclass(self, log_sink) -> None:
        # Defining the event already registers it; emit should not care
        # which subclass — only that it is a BaseEvent instance.
        emit(_PingEvent(note="generic"))
        assert any("[__PING__]" in str(m) for m, _ in log_sink)

    def test_emit_rejects_non_BaseEvent(self) -> None:
        with pytest.raises(TypeError, match="BaseEvent"):
            emit({"event": "FAKE"})  # type: ignore[arg-type]

    def test_emit_rejects_abstract_BaseEvent_without_event_name(self) -> None:
        # BaseEvent itself has event="" — emit must refuse to log an
        # unnamed event (indicates a programming error: caller instantiated
        # the abstract base instead of a concrete subclass).
        with pytest.raises(ValueError, match="event name"):
            emit(BaseEvent())
