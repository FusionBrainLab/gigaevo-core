"""Tests for the non-blocking logging setup.

The regression under test: with loguru ``enqueue=True`` the event-loop thread's
``logger.*`` call could block in ``pipe_write`` when the NFS-backed sink stalled,
freezing the whole run. The sink must instead DROP records under back-pressure and
never block the caller.
"""

from __future__ import annotations

import os
import threading
import time

from loguru import logger


def test_nonblocking_sink_drops_instead_of_blocking_when_destination_stalls():
    from gigaevo.utils.logger_setup import _NonBlockingSink

    release = threading.Event()
    started = threading.Event()

    def stalled_write(_text: str) -> None:
        started.set()
        release.wait()  # emulate an NFS write / rotation that never returns

    sink = _NonBlockingSink(stalled_write, name="test-stall", maxsize=8)
    try:
        sink(
            "first\n"
        )  # pulled by the drain thread, which then blocks in stalled_write
        assert started.wait(timeout=2.0), (
            "drain thread never reached the (stalled) destination"
        )

        t0 = time.monotonic()
        for i in range(10000):
            sink(f"line{i}\n")
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0, (
            f"producer blocked for {elapsed:.2f}s — sink is not non-blocking"
        )
        assert sink.dropped > 0, (
            "a stalled destination must cause drops, not back-pressure"
        )
    finally:
        release.set()


def test_rotating_file_writer_rotates_and_stays_bounded(tmp_path):
    from gigaevo.utils.logger_setup import _RotatingFileWriter

    path = tmp_path / "run.log"
    writer = _RotatingFileWriter(str(path), max_bytes=200)
    for _ in range(100):
        writer("x" * 50 + "\n")  # ~5100 bytes total, far past max_bytes

    assert (tmp_path / "run.log.1").exists(), (
        "rotation must keep one previous generation"
    )
    assert path.stat().st_size < 400, (
        "current file must stay near max_bytes, not grow unbounded"
    )


def test_nonblocking_sink_flush_drains_queued_records_within_timeout():
    from gigaevo.utils.logger_setup import _NonBlockingSink

    written: list[str] = []
    sink = _NonBlockingSink(written.append, name="test-flush", maxsize=1000)
    for i in range(200):
        sink(f"line{i}\n")

    sink.flush(timeout=2.0)

    assert len(written) == 200, "flush must fully drain the queued records"


def test_nonblocking_sink_flush_is_bounded_when_destination_stalls():
    from gigaevo.utils.logger_setup import _NonBlockingSink

    release = threading.Event()
    started = threading.Event()

    def stalled_write(_text: str) -> None:
        started.set()
        release.wait()  # a destination that never returns

    sink = _NonBlockingSink(stalled_write, name="test-flush-stall", maxsize=8)
    try:
        sink("first\n")
        assert started.wait(timeout=2.0)
        t0 = time.monotonic()
        sink.flush(timeout=0.2)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, (
            f"flush blocked for {elapsed:.2f}s — it must be bounded, not join a "
            "stalled writer"
        )
    finally:
        release.set()


def test_nonblocking_sink_emits_marker_after_dropping_under_backpressure():
    from gigaevo.utils.logger_setup import _NonBlockingSink

    release = threading.Event()
    started = threading.Event()
    written: list[str] = []

    def gated_write(text: str) -> None:
        started.set()
        release.wait()  # hold the drain thread so the queue backs up and drops
        written.append(text)

    sink = _NonBlockingSink(gated_write, name="test-drop-marker", maxsize=4)
    try:
        sink("first\n")  # pulled by the drain thread, which then blocks
        assert started.wait(timeout=2.0)
        for i in range(1000):
            sink(f"line{i}\n")  # overflow the bounded queue → drops
        assert sink.dropped > 0
    finally:
        release.set()

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if any("dropped" in text and "back-pressure" in text for text in written):
            break
        time.sleep(0.02)
    assert any("dropped" in text and "back-pressure" in text for text in written), (
        "a back-pressure drop must surface a marker once the queue recovers"
    )


def test_rotating_file_writer_keeps_logging_when_rotation_fails(tmp_path, monkeypatch):
    from gigaevo.utils import logger_setup

    path = tmp_path / "run.log"
    writer = logger_setup._RotatingFileWriter(str(path), max_bytes=100)

    def failing_replace(_src, _dst):
        raise OSError("simulated NFS ESTALE on rename")

    monkeypatch.setattr(logger_setup.os, "replace", failing_replace)

    for _ in range(5):
        writer("y" * 60 + "\n")  # each write trips the (now-failing) rotation
    writer("SENTINEL-AFTER-FAILED-ROTATE\n")

    assert "SENTINEL-AFTER-FAILED-ROTATE" in path.read_text(), (
        "a failed os.replace must not leave the log handle closed for the rest "
        "of the run"
    )


def test_rotating_file_writer_self_heals_from_a_closed_handle(tmp_path):
    from gigaevo.utils.logger_setup import _RotatingFileWriter

    path = tmp_path / "run.log"
    writer = _RotatingFileWriter(str(path), max_bytes=0)  # no rotation in play
    writer._fh.close()  # a prior rotation whose own reopen failed → dead handle

    writer("SENTINEL-AFTER-CLOSED-HANDLE\n")

    assert "SENTINEL-AFTER-CLOSED-HANDLE" in path.read_text(), (
        "a write against a closed handle must reopen and land, not die silently "
        "for the rest of the run once NFS recovers"
    )


def test_parse_size_handles_sizes_and_ignores_time_policies():
    from gigaevo.utils.logger_setup import _parse_size

    assert _parse_size("50 MB") == 50 * 1024**2
    assert _parse_size("100MB") == 100 * 1024**2
    assert _parse_size("1 GB") == 1024**3
    assert _parse_size("512 KB") == 512 * 1024
    assert _parse_size("1 day") == 0  # time-based rotation is not a byte size
    assert _parse_size("") == 0


def test_setup_logger_creates_file_and_records_land(tmp_path):
    from gigaevo.utils.logger_setup import setup_logger

    try:
        path = setup_logger(log_dir=str(tmp_path), level="DEBUG", enable_colors=False)
        assert os.path.exists(path)

        logger.info("hello-smoke-marker-42")
        deadline = time.monotonic() + 3.0
        found = False
        while time.monotonic() < deadline:  # drain thread flushes asynchronously
            with open(path, encoding="utf-8", errors="ignore") as fh:
                if "hello-smoke-marker-42" in fh.read():
                    found = True
                    break
            time.sleep(0.05)
        assert found, "a logged line never reached the file sink"
    finally:
        logger.remove()
