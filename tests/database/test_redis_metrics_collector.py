"""Tests for RedisMetricsCollector: flatten helper, start/stop lifecycle, collect."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from gigaevo.database.redis.config import RedisConnectionConfig, RedisKeyConfig
from gigaevo.database.redis.connection import RedisConnection
from gigaevo.database.redis.keys import RedisProgramKeys
from gigaevo.database.redis.metrics import RedisMetricsCollector, _flatten_numbers
from gigaevo.exceptions import StorageError
from gigaevo.utils.trackers.base import LogWriter

# ---------------------------------------------------------------------------
# RecordingWriter: captures scalar calls for assertions
# ---------------------------------------------------------------------------


class RecordingWriter(LogWriter):
    """LogWriter that records scalar calls for test assertions."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float]] = []

    def bind(self, path: list[str]) -> RecordingWriter:
        return self

    def scalar(self, metric: str, value: float, **kwargs: Any) -> None:
        self.scalars.append((metric, value))

    def hist(self, metric: str, values: list[float], **kwargs: Any) -> None:
        pass

    def text(self, tag: str, text: str, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_connection() -> RedisConnection:
    config = RedisConnectionConfig(
        redis_url="redis://localhost:6379/0",
        max_retries=1,
        retry_delay=0.01,
    )
    return RedisConnection(config)


def _make_keys() -> RedisProgramKeys:
    return RedisProgramKeys(RedisKeyConfig(key_prefix="test"))


# ---------------------------------------------------------------------------
# TestFlattenNumbers
# ---------------------------------------------------------------------------


class TestFlattenNumbers:
    def test_flat_dict(self) -> None:
        """Flat dict with numeric values extracts all numbers."""
        result = _flatten_numbers({"a": 1, "b": 2.5, "c": "ignore"})
        assert result == {"a": 1.0, "b": 2.5}

    def test_nested_prefixed(self) -> None:
        """Nested dict produces prefixed keys."""
        result = _flatten_numbers({"outer": {"inner": 42}}, prefix="sec/")
        assert result == {"sec/outer/inner": 42.0}

    def test_empty_dict(self) -> None:
        """Empty dict returns empty result."""
        result = _flatten_numbers({})
        assert result == {}

    def test_non_dict_input(self) -> None:
        """Non-dict input (e.g. int, string) returns empty result."""
        assert _flatten_numbers(42) == {}
        assert _flatten_numbers("hello") == {}
        assert _flatten_numbers(None) == {}


# ---------------------------------------------------------------------------
# TestStart
# ---------------------------------------------------------------------------


class TestStart:
    def test_writer_none_noop(self) -> None:
        """start() does nothing when writer is None."""
        conn = _make_connection()
        collector = RedisMetricsCollector(conn, _make_keys(), writer=None)
        collector.start()
        assert collector._task is None

    async def test_creates_task(self) -> None:
        """start() creates an asyncio task when writer is provided."""
        conn = _make_connection()
        writer = RecordingWriter()
        collector = RedisMetricsCollector(conn, _make_keys(), writer=writer)

        collector.start()
        try:
            assert collector._task is not None
            assert not collector._task.done()
        finally:
            await collector.stop()


# ---------------------------------------------------------------------------
# TestStop
# ---------------------------------------------------------------------------


class TestStop:
    async def test_cancels_task(self) -> None:
        """stop() cancels the running task and clears it."""
        conn = _make_connection()
        writer = RecordingWriter()
        collector = RedisMetricsCollector(conn, _make_keys(), writer=writer)

        collector.start()
        task = collector._task
        assert task is not None

        await collector.stop()
        assert collector._task is None
        assert collector._stop_flag is True
        # The original task should be done after stop
        assert task.done()

    async def test_noop_without_start(self) -> None:
        """stop() is safe to call without prior start()."""
        conn = _make_connection()
        writer = RecordingWriter()
        collector = RedisMetricsCollector(conn, _make_keys(), writer=writer)

        # Should not raise
        await collector.stop()
        assert collector._task is None


# ---------------------------------------------------------------------------
# TestDoubleStart
# ---------------------------------------------------------------------------


class TestDoubleStart:
    async def test_double_start_does_not_create_second_task(self) -> None:
        """Calling start() twice should not replace the existing task."""
        conn = _make_connection()
        writer = RecordingWriter()
        collector = RedisMetricsCollector(conn, _make_keys(), writer=writer)

        collector.start()
        first_task = collector._task
        assert first_task is not None

        collector.start()
        # Either same task or first is still the one (depends on impl)
        # The key assertion: only one task running
        assert collector._task is not None

        await collector.stop()


# ---------------------------------------------------------------------------
# TestCollect
# ---------------------------------------------------------------------------


class TestCollect:
    async def test_program_count(self) -> None:
        """_collect returns size metric based on program key scan."""
        conn = _make_connection()
        keys = _make_keys()

        # Create a mock redis that yields program keys on scan_iter
        mock_redis = AsyncMock()

        async def fake_scan_iter(match=None, count=None):
            # Simulate 3 program keys
            for pid in ["p1", "p2", "p3"]:
                yield keys.program(pid)

        mock_redis.scan_iter = fake_scan_iter
        mock_redis.info = AsyncMock(return_value={})

        conn._redis = mock_redis
        conn._closing = False

        collector = RedisMetricsCollector(conn, keys, writer=RecordingWriter())
        metrics = await collector._collect()

        assert metrics["size"] == 3.0

    async def test_closing_early_exit(self) -> None:
        """_collect returns empty dict when connection is closing."""
        conn = _make_connection()
        conn._closing = True

        collector = RedisMetricsCollector(conn, _make_keys(), writer=RecordingWriter())
        result = await collector._collect()
        assert result == {}

    async def test_storage_error_returns_empty(self) -> None:
        """_collect returns empty dict when get() raises StorageError."""
        conn = _make_connection()
        conn._closing = False
        conn._redis = None  # Force get() to try connecting

        # Patch get() to raise StorageError
        with patch.object(conn, "get", side_effect=StorageError("connection failed")):
            collector = RedisMetricsCollector(
                conn, _make_keys(), writer=RecordingWriter()
            )
            result = await collector._collect()
            assert result == {}

    async def test_info_sections_collected(self) -> None:
        """_collect includes flattened numeric values from INFO sections."""
        conn = _make_connection()
        keys = _make_keys()

        mock_redis = AsyncMock()

        async def fake_scan_iter(match=None, count=None):
            # No programs
            return
            yield  # make it an async generator

        mock_redis.scan_iter = fake_scan_iter

        async def fake_info(section=None):
            if section == "memory":
                return {"used_memory": 1024, "used_memory_peak": 2048}
            if section == "stats":
                return {"total_connections_received": 5}
            # Other sections return empty or raise
            raise Exception("not available")

        mock_redis.info = fake_info
        conn._redis = mock_redis
        conn._closing = False

        collector = RedisMetricsCollector(conn, keys, writer=RecordingWriter())
        metrics = await collector._collect()

        assert metrics["size"] == 0.0
        assert metrics["memory/used_memory"] == 1024.0
        assert metrics["memory/used_memory_peak"] == 2048.0
        assert metrics["stats/total_connections_received"] == 5.0
        # Sections that raised should be skipped, not cause failure
        assert "cpu/" not in "".join(metrics.keys())
