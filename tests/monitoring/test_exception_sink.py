"""Tests for the EXCEPTION canonical-event sink hook.

`install_exception_sink()` is the single seam: any call to `logger.exception(...)`
must produce exactly one `[EXCEPTION] {json}` line.

Critically, the sink must NOT re-fire on canonical event lines — otherwise
the EXCEPTION emitted by the sink would be seen by the sink, causing
infinite recursion. The sink rejects its own emissions via
`record["extra"]["canonical_event"] == True`.

The sink delivers on a background thread, so assertions poll with a bounded
wait rather than reading the capture list straight after the log call.
"""

from __future__ import annotations

import json
import re
import threading
import time

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


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestExceptionSink:
    def test_logger_exception_emits_single_exception_event(
        self, capture_sink, exception_sink_installed
    ):
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("something failed")

        assert _wait_for(lambda: _exc_lines(capture_sink)), "no EXCEPTION event emitted"
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
        assert not _wait_for(lambda: _exc_lines(capture_sink), timeout=0.5)

    def test_ordinary_logs_do_not_block_when_emit_stalls(
        self, capture_sink, exception_sink_installed, monkeypatch
    ):
        """Ordinary (non-exception) logging must never reach the sink's queue.

        A record handed to the sink can only be drained as fast as `emit()`
        runs; anything queued behind a stalled `emit()` is a freeze risk for the
        producer, which in production is the asyncio event loop. Gating inside
        the sink body is too late — the record is already queued. Stall the
        drain and flood with ordinary DEBUG records: they must be rejected
        before the queue, so the caller never blocks.
        """
        drain_entered = threading.Event()
        release = threading.Event()

        def _stalled_emit(_event):
            drain_entered.set()
            release.wait(timeout=30)

        monkeypatch.setattr(
            "gigaevo.monitoring.exception_sink.emit", _stalled_emit, raising=True
        )
        try:
            # Occupy the sink's drain thread so nothing more can leave the queue.
            try:
                raise ValueError("stall")
            except ValueError:
                logger.exception("stall the drain thread")
            assert drain_entered.wait(timeout=10), "sink drain never ran"

            done = threading.Event()

            def _flood():
                for i in range(20_000):
                    logger.debug(f"filler {i}")
                done.set()

            threading.Thread(target=_flood, daemon=True).start()
            assert done.wait(timeout=20), (
                "logger.debug() blocked: ordinary records are reaching the "
                "exception sink's queue"
            )
        finally:
            release.set()

    def test_bound_exc_type_is_used_when_no_active_exception(
        self, capture_sink, exception_sink_installed
    ):
        """A caller handling a *returned* failure can name the exception class.

        `logger.exception(...)` only populates the record's exception tuple when
        called from inside an `except` block. Callers that report a failure
        object instead (a stage result carrying its own error) have the class
        name in hand — the sink must use it rather than reporting "Unknown".
        """
        logger.bind(exc_type="QhullError").exception("stage failed")

        assert _wait_for(lambda: _exc_lines(capture_sink)), "no EXCEPTION event emitted"
        body = json.loads(re.search(r"\{.*\}$", _exc_lines(capture_sink)[0]).group(0))
        assert body["exc_type"] == "QhullError"

    def test_live_exception_type_wins_without_binding(
        self, capture_sink, exception_sink_installed
    ):
        """The unbound path must keep reading the live exception tuple."""
        try:
            raise KeyError("missing")
        except KeyError:
            logger.exception("raised for real")

        assert _wait_for(lambda: _exc_lines(capture_sink)), "no EXCEPTION event emitted"
        body = json.loads(re.search(r"\{.*\}$", _exc_lines(capture_sink)[0]).group(0))
        assert body["exc_type"] == "KeyError"

    def test_exc_type_unknown_only_when_nothing_is_available(
        self, capture_sink, exception_sink_installed
    ):
        """No active exception and no bound type — "Unknown" is the honest answer."""
        logger.exception("nothing to introspect")

        assert _wait_for(lambda: _exc_lines(capture_sink)), "no EXCEPTION event emitted"
        body = json.loads(re.search(r"\{.*\}$", _exc_lines(capture_sink)[0]).group(0))
        assert body["exc_type"] == "Unknown"

    def test_sink_does_not_recurse_on_canonical_events(
        self, capture_sink, exception_sink_installed
    ):
        """Emitting a canonical event with extra.canonical_event=True from
        inside the sink's scope must not re-trigger the sink (no recursion).
        """
        logger.bind(canonical_event=True).info('[EXCEPTION] {"fake":true}')
        # The captured line IS the forged canonical event; the sink must not
        # add a second one of its own.
        assert not _wait_for(lambda: len(_exc_lines(capture_sink)) > 1, timeout=0.5)
        assert len(_exc_lines(capture_sink)) == 1
