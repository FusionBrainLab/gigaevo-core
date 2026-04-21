"""Tests for the EXCEPTION canonical-event sink hook.

`install_exception_sink()` is the single seam: any call to `logger.exception(...)`
must produce exactly one `[EXCEPTION] {json}` line.

Critically, the sink must NOT re-fire on canonical event lines — otherwise
the EXCEPTION emitted by the sink would be seen by the sink, causing
infinite recursion. The sink detects its own emissions via
`record["extra"]["canonical_event"] == True` and skips them.
"""

from __future__ import annotations

import json
import re

from loguru import logger
import pytest

from gigaevo.monitoring.exception_sink import install_exception_sink


@pytest.fixture
def capture_sink():
    captured: list[str] = []

    def sink(message):
        captured.append(str(message))

    cap_id = logger.add(sink, level="DEBUG", format="{message}")
    yield captured
    logger.remove(cap_id)


@pytest.fixture
def exception_sink_installed():
    sink_id = install_exception_sink()
    yield
    logger.remove(sink_id)


def _exc_lines(captured):
    return [m for m in captured if "[EXCEPTION]" in m]


class TestExceptionSink:
    def test_logger_exception_emits_single_exception_event(
        self, capture_sink, exception_sink_installed
    ):
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("something failed")

        # Exception sink runs on a background thread (enqueue=True) to avoid
        # loguru's _protected_lock re-entry when emit() calls logger.info().
        # Flush the queue before reading capture_sink.
        logger.complete()

        lines = _exc_lines(capture_sink)
        assert len(lines) == 1, f"expected exactly one EXCEPTION line, got {lines}"
        body = json.loads(re.search(r"\{.*\}$", lines[0]).group(0))
        assert body["event"] == "EXCEPTION"
        assert body["exc_type"] == "ValueError"
        assert "something failed" in body["msg_head"]

    def test_non_exception_logs_do_not_emit(
        self, capture_sink, exception_sink_installed
    ):
        logger.info("just info")
        logger.warning("just warning")
        logger.complete()
        lines = _exc_lines(capture_sink)
        assert lines == []

    def test_sink_does_not_recurse_on_canonical_events(
        self, capture_sink, exception_sink_installed
    ):
        """Emitting a canonical event with extra.canonical_event=True from
        inside the sink's scope must not re-trigger the sink (no recursion).
        """
        logger.bind(canonical_event=True).info('[EXCEPTION] {"fake":true}')
        logger.complete()
        # The captured line IS the forged canonical event, but the sink did
        # NOT emit a fresh EXCEPTION event of its own.
        captured_event_lines = [m for m in capture_sink if "[EXCEPTION]" in m]
        # The forged line itself counts as one; there must not be a second
        # one produced by the sink.
        assert len(captured_event_lines) == 1
