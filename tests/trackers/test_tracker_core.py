"""Tests for GenericLogger / BoundGeneric: bind, scalar, _offer, _state, throttle.

Covers untested tracker internals:
- bind (6 production callers)
- scalar (6 production callers)
- _offer (3 production callers)
- _state (2 production callers)
- _throttle (dedup logic)
- BoundGeneric path chaining
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
# In-memory backend for testing
# ---------------------------------------------------------------------------


class MemoryBackend(LoggerBackend):
    """Collects all writes in memory for assertions."""

    def __init__(self):
        self.scalars: list[tuple[str, float, int, float]] = []
        self.hists: list[tuple[str, Any, int, float]] = []
        self.texts: list[tuple[str, str, int, float]] = []
        self.flush_count = 0

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def write_scalar(self, tag: str, value: float, step: int, wall_time: float) -> None:
        self.scalars.append((tag, value, step, wall_time))

    def write_hist(self, tag: str, values: Any, step: int, wall_time: float) -> None:
        self.hists.append((tag, values, step, wall_time))

    def write_text(self, tag: str, text: str, step: int, wall_time: float) -> None:
        self.texts.append((tag, text, step, wall_time))

    def flush(self) -> None:
        self.flush_count += 1


@pytest.fixture
def backend():
    return MemoryBackend()


@pytest.fixture
def logger_instance(backend):
    lg = GenericLogger(backend, queue_size=1024, flush_secs=999)
    yield lg
    lg.close()


# ===================================================================
# _sanitize / _render_tag
# ===================================================================


class TestSanitize:
    def test_alphanumeric_unchanged(self):
        assert _sanitize("hello123") == "hello123"

    def test_special_chars_replaced(self):
        assert _sanitize("a/b c:d") == "a_b_c_d"

    def test_allowed_chars_preserved(self):
        assert _sanitize("a-b_c.d") == "a-b_c.d"


class TestRenderTag:
    def test_simple(self):
        assert _render_tag(["llm", "tokens"], "total") == "llm/tokens/total"

    def test_empty_path(self):
        assert _render_tag([], "metric") == "metric"

    def test_sanitizes_components(self):
        result = _render_tag(["path with spaces"], "metric/name")
        assert " " not in result


# ===================================================================
# GenericLogger.scalar / bind / close
# ===================================================================


class TestGenericLogger:
    def test_scalar_writes_to_backend(self, logger_instance, backend):
        """scalar() should eventually reach the backend."""
        logger_instance.scalar("loss", 0.5)
        # Give the writer thread time to process
        logger_instance.close()
        assert len(backend.scalars) == 1
        assert backend.scalars[0][0] == "loss"
        assert backend.scalars[0][1] == 0.5

    def test_scalar_auto_increments_step(self, logger_instance, backend):
        """Sequential scalars without explicit step get auto-incremented."""
        logger_instance.scalar("loss", 0.5)
        logger_instance.scalar("loss", 0.4)
        logger_instance.scalar("loss", 0.3)
        logger_instance.close()
        steps = [s[2] for s in backend.scalars]
        assert steps == [0, 1, 2]

    def test_scalar_explicit_step(self, logger_instance, backend):
        """Explicit step overrides auto-increment."""
        logger_instance.scalar("loss", 0.5, step=10)
        logger_instance.close()
        assert backend.scalars[0][2] == 10

    def test_hist_writes_to_backend(self, logger_instance, backend):
        logger_instance.hist("weights", [1.0, 2.0, 3.0])
        logger_instance.close()
        assert len(backend.hists) == 1

    def test_text_writes_to_backend(self, logger_instance, backend):
        logger_instance.text("summary", "hello world")
        logger_instance.close()
        assert len(backend.texts) == 1
        assert backend.texts[0][1] == "hello world"

    def test_closed_logger_drops_events(self, logger_instance, backend):
        """After close(), new events are silently dropped."""
        logger_instance.close()
        logger_instance.scalar("loss", 999.0)
        assert not any(s[1] == 999.0 for s in backend.scalars)

    def test_double_close_safe(self, logger_instance):
        """Closing twice should not raise."""
        logger_instance.close()
        logger_instance.close()


# ===================================================================
# BoundGeneric (path chaining)
# ===================================================================


class TestBoundGeneric:
    def test_bind_creates_bound(self, logger_instance):
        bound = logger_instance.bind(path=["llm", "tokens"])
        assert isinstance(bound, BoundGeneric)

    def test_bound_scalar_prepends_path(self, logger_instance, backend):
        bound = logger_instance.bind(path=["llm", "tokens"])
        bound.scalar("total", 100.0)
        logger_instance.close()
        assert len(backend.scalars) == 1
        assert backend.scalars[0][0] == "llm/tokens/total"

    def test_nested_bind(self, logger_instance, backend):
        """Nested bind() chains paths."""
        bound1 = logger_instance.bind(path=["llm"])
        bound2 = bound1.bind(path=["tokens"])
        bound2.scalar("total", 50.0)
        logger_instance.close()
        assert backend.scalars[0][0] == "llm/tokens/total"

    def test_bound_hist(self, logger_instance, backend):
        bound = logger_instance.bind(path=["metrics"])
        bound.hist("dist", [1.0, 2.0])
        logger_instance.close()
        assert backend.hists[0][0] == "metrics/dist"

    def test_bound_text(self, logger_instance, backend):
        bound = logger_instance.bind(path=["logs"])
        bound.text("event", "something happened")
        logger_instance.close()
        assert backend.texts[0][0] == "logs/event"

    def test_bound_close_closes_parent(self, logger_instance, backend):
        bound = logger_instance.bind(path=["sub"])
        bound.close()
        # Parent should be closed
        assert logger_instance._closed


# ===================================================================
# _offer (queue behavior)
# ===================================================================


class TestOffer:
    def test_full_queue_does_not_raise(self):
        """When queue is full, _offer silently drops instead of blocking."""
        backend = MemoryBackend()
        lg = GenericLogger(backend, queue_size=1, flush_secs=999)
        # Fill queue rapidly — should not raise
        for _ in range(100):
            lg._offer({"k": "scalar", "metric": "x", "value": 0.0, "path": []})
        lg.close()


# ===================================================================
# _state (lazy initialization)
# ===================================================================


class TestState:
    def test_creates_new_state(self, logger_instance):
        st = logger_instance._state("new_key")
        assert st.last_step == -1
        assert st.last_value is None

    def test_returns_same_state(self, logger_instance):
        st1 = logger_instance._state("key")
        st2 = logger_instance._state("key")
        assert st1 is st2


# ===================================================================
# _throttle (deduplication)
# ===================================================================


class TestThrottle:
    def test_no_throttle_by_default(self, logger_instance):
        """With empty throttle config, nothing is throttled."""
        assert logger_instance._throttle("k", 1.0, time.time(), {}) is False

    def test_min_interval_throttles(self, logger_instance):
        """Events within min_interval_s are throttled."""
        now = time.time()
        assert logger_instance._throttle("k", 1.0, now, {"min_interval_s": 10}) is False
        assert (
            logger_instance._throttle("k", 2.0, now + 0.001, {"min_interval_s": 10})
            is True
        )

    def test_min_interval_allows_after_delay(self, logger_instance):
        """Events after min_interval_s passes are allowed."""
        now = time.time()
        assert logger_instance._throttle("k", 1.0, now, {"min_interval_s": 1}) is False
        assert (
            logger_instance._throttle("k", 2.0, now + 2, {"min_interval_s": 1}) is False
        )

    def test_min_delta_throttles_small_changes(self, logger_instance):
        """Changes smaller than min_delta are throttled."""
        now = time.time()
        assert logger_instance._throttle("k", 1.0, now, {"min_delta": 0.5}) is False
        assert logger_instance._throttle("k", 1.1, now + 1, {"min_delta": 0.5}) is True

    def test_min_delta_allows_large_changes(self, logger_instance):
        """Changes larger than min_delta are allowed."""
        now = time.time()
        assert logger_instance._throttle("k", 1.0, now, {"min_delta": 0.5}) is False
        assert logger_instance._throttle("k", 2.0, now + 1, {"min_delta": 0.5}) is False
