"""Tests for GenericLogger and BoundGeneric (trackers/core.py).

Covers the threaded writer, throttling, step tracking, path binding,
and BoundGeneric delegation — all via a synchronous RecordingBackend.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from gigaevo.utils.trackers.core import (
    BoundGeneric,
    GenericLogger,
    LoggerBackend,
    _render_tag,
    _sanitize,
)

# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestSanitize:
    def test_alphanumeric_unchanged(self):
        assert _sanitize("hello123") == "hello123"

    def test_allowed_chars_preserved(self):
        assert _sanitize("hello-world_v1.2") == "hello-world_v1.2"

    def test_space_replaced(self):
        assert _sanitize("a b") == "a_b"

    def test_slash_replaced(self):
        assert _sanitize("a/b") == "a_b"

    def test_empty_string(self):
        assert _sanitize("") == ""

    def test_non_string_coerced(self):
        result = _sanitize(123)
        assert result == "123"

    def test_all_special_chars_replaced(self):
        result = _sanitize("a!b@c#d")
        assert result == "a_b_c_d"


class TestRenderTag:
    def test_path_and_metric(self):
        assert _render_tag(["a", "b"], "loss") == "a/b/loss"

    def test_empty_path(self):
        assert _render_tag([], "acc") == "acc"

    def test_empty_strings_filtered(self):
        assert _render_tag(["", "b"], "loss") == "b/loss"

    def test_sanitizes_chars(self):
        assert _render_tag(["my path"], "my metric") == "my_path/my_metric"

    def test_single_element_path(self):
        assert _render_tag(["train"], "loss") == "train/loss"


# ---------------------------------------------------------------------------
# Recording backend — no real I/O
# ---------------------------------------------------------------------------


class RecordingBackend(LoggerBackend):
    """Backend that records all calls for assertion."""

    def __init__(self):
        self.opened = False
        self.closed = False
        self.flushed = 0
        self.scalars: list[tuple] = []
        self.hists: list[tuple] = []
        self.texts: list[tuple] = []

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def flush(self) -> None:
        self.flushed += 1

    def write_scalar(self, tag: str, value: float, step: int, wall_time: float) -> None:
        self.scalars.append((tag, value, step, wall_time))

    def write_hist(self, tag: str, values: Any, step: int, wall_time: float) -> None:
        self.hists.append((tag, values, step, wall_time))

    def write_text(self, tag: str, text: str, step: int, wall_time: float) -> None:
        self.texts.append((tag, text, step, wall_time))


def _make_logger(flush_secs: float = 60.0) -> tuple[GenericLogger, RecordingBackend]:
    backend = RecordingBackend()
    logger = GenericLogger(backend, queue_size=256, flush_secs=flush_secs)
    return logger, backend


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestGenericLoggerInit:
    def test_backend_opened_on_init(self):
        backend = RecordingBackend()
        GenericLogger(backend).close()
        assert backend.opened

    def test_worker_thread_started(self):
        logger, _ = _make_logger()
        assert logger._t.is_alive()
        logger.close()


# ---------------------------------------------------------------------------
# scalar
# ---------------------------------------------------------------------------


class TestGenericLoggerScalar:
    def test_scalar_reaches_backend(self):
        logger, backend = _make_logger()
        logger.scalar("loss", 0.5)
        logger.close()
        assert len(backend.scalars) == 1
        tag, value, step, _ = backend.scalars[0]
        assert tag == "loss"
        assert value == pytest.approx(0.5)
        assert step == 0  # first call → auto-step 0

    def test_scalar_auto_increments_step(self):
        logger, backend = _make_logger()
        logger.scalar("loss", 1.0)
        logger.scalar("loss", 2.0)
        logger.close()
        steps = [s[2] for s in backend.scalars]
        assert steps == [0, 1]

    def test_scalar_explicit_step(self):
        logger, backend = _make_logger()
        logger.scalar("loss", 1.0, step=42)
        logger.close()
        assert backend.scalars[0][2] == 42

    def test_scalar_with_path(self):
        logger, backend = _make_logger()
        logger.scalar("loss", 1.0, path=["train"])
        logger.close()
        assert backend.scalars[0][0] == "train/loss"

    def test_no_write_after_close(self):
        logger, backend = _make_logger()
        logger.close()
        logger.scalar("loss", 9.9)  # silently dropped
        assert len(backend.scalars) == 0

    def test_different_metrics_have_independent_steps(self):
        logger, backend = _make_logger()
        logger.scalar("a", 1.0)
        logger.scalar("b", 2.0)
        logger.scalar("a", 3.0)
        logger.close()
        a_steps = [s[2] for s in backend.scalars if s[0] == "a"]
        b_steps = [s[2] for s in backend.scalars if s[0] == "b"]
        assert a_steps == [0, 1]
        assert b_steps == [0]


# ---------------------------------------------------------------------------
# hist
# ---------------------------------------------------------------------------


class TestGenericLoggerHist:
    def test_hist_reaches_backend(self):
        logger, backend = _make_logger()
        logger.hist("weights", [1.0, 2.0, 3.0])
        logger.close()
        assert len(backend.hists) == 1
        tag, values, step, _ = backend.hists[0]
        assert tag == "weights"
        assert values == [1.0, 2.0, 3.0]

    def test_hist_auto_step(self):
        logger, backend = _make_logger()
        logger.hist("w", [1.0])
        logger.hist("w", [2.0])
        logger.close()
        assert backend.hists[0][2] == 0
        assert backend.hists[1][2] == 1

    def test_hist_with_path(self):
        logger, backend = _make_logger()
        logger.hist("dist", [1.0], path=["eval"])
        logger.close()
        assert backend.hists[0][0] == "eval/dist"


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------


class TestGenericLoggerText:
    def test_text_reaches_backend(self):
        logger, backend = _make_logger()
        logger.text("log", "hello world")
        logger.close()
        assert len(backend.texts) == 1
        tag, text, step, _ = backend.texts[0]
        assert tag == "log"
        assert text == "hello world"

    def test_text_with_path(self):
        logger, backend = _make_logger()
        logger.text("msg", "hi", path=["debug"])
        logger.close()
        assert backend.texts[0][0] == "debug/msg"


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------


class TestGenericLoggerThrottle:
    def test_min_delta_suppresses_small_change(self):
        logger, backend = _make_logger()
        logger.scalar("loss", 1.0, throttle={"min_delta": 0.5})
        logger.scalar("loss", 1.1, throttle={"min_delta": 0.5})  # |1.1-1.0|=0.1 < 0.5
        logger.close()
        assert len(backend.scalars) == 1
        assert backend.scalars[0][1] == pytest.approx(1.0)

    def test_min_delta_passes_large_change(self):
        logger, backend = _make_logger()
        logger.scalar("loss", 1.0, throttle={"min_delta": 0.5})
        logger.scalar("loss", 2.0, throttle={"min_delta": 0.5})  # |2.0-1.0|=1.0 >= 0.5
        logger.close()
        assert len(backend.scalars) == 2

    def test_min_interval_suppresses_fast_update(self):
        logger, backend = _make_logger()
        now = time.time()
        logger.scalar("loss", 1.0, wall_time=now, throttle={"min_interval_s": 60.0})
        logger.scalar("loss", 2.0, wall_time=now + 1, throttle={"min_interval_s": 60.0})
        logger.close()
        assert len(backend.scalars) == 1

    def test_min_interval_passes_after_enough_time(self):
        logger, backend = _make_logger()
        now = time.time()
        logger.scalar("loss", 1.0, wall_time=now, throttle={"min_interval_s": 5.0})
        logger.scalar("loss", 2.0, wall_time=now + 10, throttle={"min_interval_s": 5.0})
        logger.close()
        assert len(backend.scalars) == 2

    def test_no_throttle_all_pass(self):
        logger, backend = _make_logger()
        for i in range(5):
            logger.scalar("loss", float(i))
        logger.close()
        assert len(backend.scalars) == 5

    def test_first_call_always_passes_min_delta(self):
        """First value has no previous → no suppression."""
        logger, backend = _make_logger()
        logger.scalar("loss", 99.0, throttle={"min_delta": 1000.0})
        logger.close()
        assert len(backend.scalars) == 1


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestGenericLoggerClose:
    def test_close_is_idempotent(self):
        logger, backend = _make_logger()
        logger.close()
        logger.close()  # should not raise
        assert backend.closed

    def test_close_calls_flush_and_close_on_backend(self):
        logger, backend = _make_logger()
        logger.close()
        assert backend.flushed >= 1
        assert backend.closed

    def test_worker_thread_stops_after_close(self):
        logger, _ = _make_logger()
        logger.close()
        logger._t.join(timeout=3.0)
        assert not logger._t.is_alive()


# ---------------------------------------------------------------------------
# bind / BoundGeneric
# ---------------------------------------------------------------------------


class TestGenericLoggerBind:
    def test_bind_returns_bound_generic(self):
        logger, _ = _make_logger()
        bound = logger.bind(path=["runner"])
        assert isinstance(bound, BoundGeneric)
        logger.close()

    def test_bind_empty_path(self):
        logger, backend = _make_logger()
        bound = logger.bind()
        bound.scalar("loss", 1.0)
        logger.close()
        assert backend.scalars[0][0] == "loss"


class TestBoundGeneric:
    def test_scalar_prepends_path(self):
        logger, backend = _make_logger()
        bound = logger.bind(path=["train"])
        bound.scalar("loss", 0.5)
        logger.close()
        assert backend.scalars[0][0] == "train/loss"

    def test_nested_bind(self):
        logger, backend = _make_logger()
        bound = logger.bind(path=["a"]).bind(path=["b"])
        bound.scalar("loss", 0.5)
        logger.close()
        assert backend.scalars[0][0] == "a/b/loss"

    def test_hist_prepends_path(self):
        logger, backend = _make_logger()
        bound = logger.bind(path=["eval"])
        bound.hist("weights", [1.0])
        logger.close()
        assert backend.hists[0][0] == "eval/weights"

    def test_text_prepends_path(self):
        logger, backend = _make_logger()
        bound = logger.bind(path=["debug"])
        bound.text("msg", "hello")
        logger.close()
        assert backend.texts[0][0] == "debug/msg"

    def test_close_delegates_to_base(self):
        logger, backend = _make_logger()
        bound = logger.bind(path=["x"])
        bound.close()
        assert backend.closed

    def test_additional_path_in_call_merges(self):
        """Extra 'path' kwarg in call is appended after the bound path."""
        logger, backend = _make_logger()
        bound = logger.bind(path=["train"])
        bound.scalar("loss", 1.0, path=["epoch"])
        logger.close()
        assert backend.scalars[0][0] == "train/epoch/loss"

    def test_bound_bind_preserves_full_path(self):
        logger, backend = _make_logger()
        b1 = logger.bind(path=["x"])
        b2 = b1.bind(path=["y", "z"])
        b2.scalar("v", 1.0)
        logger.close()
        assert backend.scalars[0][0] == "x/y/z/v"
