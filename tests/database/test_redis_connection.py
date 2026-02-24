"""Tests for RedisConnection: retry logic, double-checked locking, graceful shutdown."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.database.redis.config import RedisConnectionConfig
from gigaevo.database.redis.connection import RedisConnection
from gigaevo.exceptions import StorageError


def _make_config(**overrides) -> RedisConnectionConfig:
    defaults = {
        "redis_url": "redis://localhost:6379/0",
        "max_retries": 3,
        "retry_delay": 0.01,  # fast for tests
    }
    defaults.update(overrides)
    return RedisConnectionConfig(**defaults)


# ---------------------------------------------------------------------------
# execute() — retry logic
# ---------------------------------------------------------------------------


class TestExecuteRetry:
    async def test_success_on_first_try(self) -> None:
        conn = RedisConnection(_make_config())
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        conn._redis = mock_redis

        await conn.execute("test_op", lambda r: r.get("key"))
        mock_redis.get.assert_called_once_with("key")

    async def test_retries_on_transient_failure(self) -> None:
        conn = RedisConnection(_make_config(max_retries=3))
        mock_redis = AsyncMock()
        conn._redis = mock_redis

        call_count = 0

        async def flaky_op(r):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        result = await conn.execute("test_op", flaky_op)
        assert result == "ok"
        assert call_count == 3

    async def test_raises_after_max_retries(self) -> None:
        conn = RedisConnection(_make_config(max_retries=2))
        mock_redis = AsyncMock()
        conn._redis = mock_redis

        async def always_fail(r):
            raise ConnectionError("permanent")

        with pytest.raises(StorageError, match="test_op failed"):
            await conn.execute("test_op", always_fail)

    async def test_exponential_backoff_delay(self) -> None:
        """Verify retry delays increase exponentially."""
        conn = RedisConnection(_make_config(max_retries=3, retry_delay=0.01))
        mock_redis = AsyncMock()
        conn._redis = mock_redis

        delays = []
        original_sleep = asyncio.sleep

        async def tracking_sleep(d):
            delays.append(d)
            await original_sleep(0)  # Don't actually sleep

        async def always_fail(r):
            raise ConnectionError("fail")

        with patch("gigaevo.database.redis.connection.asyncio.sleep", tracking_sleep):
            with pytest.raises(StorageError):
                await conn.execute("test_op", always_fail)

        # max_retries=3: fail on 1st (sleep 0.01), fail on 2nd (sleep 0.02), fail on 3rd (raises)
        assert len(delays) == 2
        assert delays[0] <= delays[1]  # exponential increase

    async def test_refuses_when_closing(self) -> None:
        conn = RedisConnection(_make_config())
        conn._closing = True

        with pytest.raises(StorageError, match="closing"):
            await conn.execute("test_op", lambda r: r.get("key"))


# ---------------------------------------------------------------------------
# get() — double-checked locking
# ---------------------------------------------------------------------------


class TestGetConnection:
    async def test_returns_existing_connection(self) -> None:
        conn = RedisConnection(_make_config())
        mock_redis = AsyncMock()
        conn._redis = mock_redis

        result = await conn.get()
        assert result is mock_redis

    async def test_creates_connection_on_first_call(self) -> None:
        conn = RedisConnection(_make_config())

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        with patch(
            "gigaevo.database.redis.connection.aioredis.from_url",
            return_value=mock_redis,
        ):
            result = await conn.get()

        assert result is mock_redis
        assert conn._redis is mock_redis
        mock_redis.ping.assert_called_once()

    async def test_raises_when_closing(self) -> None:
        conn = RedisConnection(_make_config())
        conn._closing = True

        with pytest.raises(StorageError, match="closing"):
            await conn.get()

    async def test_concurrent_get_creates_one_connection(self) -> None:
        """Two concurrent get() calls should create only one connection."""
        conn = RedisConnection(_make_config())
        creation_count = 0

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        def counting_from_url(*args, **kwargs):
            nonlocal creation_count
            creation_count += 1
            return mock_redis

        with patch(
            "gigaevo.database.redis.connection.aioredis.from_url",
            side_effect=counting_from_url,
        ):
            r1, r2 = await asyncio.gather(conn.get(), conn.get())

        assert r1 is r2
        assert creation_count == 1


# ---------------------------------------------------------------------------
# close() — graceful shutdown
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_sets_closing_flag(self) -> None:
        conn = RedisConnection(_make_config())
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_redis.connection_pool = MagicMock()
        mock_redis.connection_pool.disconnect = AsyncMock()
        conn._redis = mock_redis

        await conn.close()

        assert conn.is_closing is True
        assert conn._redis is None
        mock_redis.aclose.assert_called_once()

    async def test_close_idempotent(self) -> None:
        """Calling close() twice doesn't raise."""
        conn = RedisConnection(_make_config())
        await conn.close()
        await conn.close()

        assert conn.is_closing is True

    async def test_close_suppresses_aclose_errors(self) -> None:
        """Connection errors during close are suppressed."""
        conn = RedisConnection(_make_config())
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock(side_effect=ConnectionError("already closed"))
        mock_redis.connection_pool = MagicMock()
        mock_redis.connection_pool.disconnect = AsyncMock()
        conn._redis = mock_redis

        # Should not raise
        await conn.close()
        assert conn._redis is None

    async def test_get_after_close_raises(self) -> None:
        conn = RedisConnection(_make_config())
        await conn.close()

        with pytest.raises(StorageError, match="closing"):
            await conn.get()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_is_connected_false_initially(self) -> None:
        conn = RedisConnection(_make_config())
        assert conn.is_connected is False

    def test_is_connected_true_after_set(self) -> None:
        conn = RedisConnection(_make_config())
        conn._redis = AsyncMock()
        assert conn.is_connected is True

    def test_is_closing_false_initially(self) -> None:
        conn = RedisConnection(_make_config())
        assert conn.is_closing is False
