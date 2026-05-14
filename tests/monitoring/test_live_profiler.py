"""Smoke tests for the live profiler daemon.

We test the pure render-once helper (which does the I/O) and the
thread-start contract — not the actual loop timing, which would make
tests flaky on shared CI infrastructure.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

from gigaevo.monitoring.live_profiler import (
    _render_once,
    start_live_profiler,
)


def _write_minimal_log(path: Path) -> None:
    """Write a log line that flow_profiler.parse_log accepts without erroring."""
    # parse_log tolerates empty/sparse input by returning empty lists — any
    # plain log content is fine. We use a single-line stub.
    path.write_text("2026-05-13 12:00:00 | INFO | test | hello\n")


class TestRenderOnce:
    def test_writes_html_file_atomically(self, tmp_path: Path) -> None:
        log = tmp_path / "run.log"
        _write_minimal_log(log)
        html = tmp_path / "profile_live.html"

        n_prog, n_llm = _render_once(log, html, label="live")

        assert html.exists()
        assert html.read_text().lstrip().lower().startswith("<!doctype html>")
        # Atomic write should leave no .tmp residue.
        assert not (tmp_path / "profile_live.html.tmp").exists()
        assert n_prog == 0
        assert n_llm == 0

    def test_overwrites_existing_html(self, tmp_path: Path) -> None:
        log = tmp_path / "run.log"
        _write_minimal_log(log)
        html = tmp_path / "profile_live.html"
        html.write_text("STALE")

        _render_once(log, html, label="live")

        assert "STALE" not in html.read_text()


class TestStartLiveProfiler:
    def test_returns_event_and_starts_daemon_thread(self, tmp_path: Path) -> None:
        log = tmp_path / "run.log"
        _write_minimal_log(log)
        out = tmp_path / "out"

        # Long interval so the loop doesn't tick during the test — we only
        # verify the bootstrap contract (thread started, event returned).
        stop = start_live_profiler(log, out, interval_s=3600.0)
        try:
            assert isinstance(stop, threading.Event)
            assert out.exists() and out.is_dir()

            # Find our named daemon thread.
            threads = {t.name: t for t in threading.enumerate()}
            assert "live-profiler" in threads
            assert threads["live-profiler"].daemon is True
        finally:
            stop.set()

    def test_creates_out_dir_if_missing(self, tmp_path: Path) -> None:
        log = tmp_path / "run.log"
        _write_minimal_log(log)
        out = tmp_path / "nested" / "out"

        stop = start_live_profiler(log, out, interval_s=3600.0)
        try:
            assert out.exists()
        finally:
            stop.set()

    def test_loop_renders_on_first_tick(self, tmp_path: Path) -> None:
        log = tmp_path / "run.log"
        _write_minimal_log(log)
        out = tmp_path / "out"

        stop = start_live_profiler(
            log, out, interval_s=3600.0, filename="profile_live.html"
        )
        try:
            html = out / "profile_live.html"
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not html.exists():
                time.sleep(0.05)
            assert html.exists(), "live profiler did not render within 5s"
        finally:
            stop.set()

    def test_waits_for_log_to_appear(self, tmp_path: Path) -> None:
        log = tmp_path / "not_yet.log"
        out = tmp_path / "out"

        stop = start_live_profiler(log, out, interval_s=3600.0)
        try:
            html = out / "profile_live.html"
            # Briefly: no render yet because log doesn't exist.
            time.sleep(0.3)
            assert not html.exists()

            _write_minimal_log(log)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not html.exists():
                time.sleep(0.1)
            assert html.exists()
        finally:
            stop.set()


@pytest.mark.parametrize("interval", [0.0, 1.0, 60.0])
def test_start_accepts_numeric_intervals(tmp_path: Path, interval: float) -> None:
    log = tmp_path / "run.log"
    _write_minimal_log(log)
    out = tmp_path / "out"
    stop = start_live_profiler(log, out, interval_s=interval)
    try:
        assert isinstance(stop, threading.Event)
    finally:
        stop.set()
