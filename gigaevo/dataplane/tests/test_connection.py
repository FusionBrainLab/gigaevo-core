"""Tests for :class:`gigaevo.dataplane.connection.RedisConnection`.

Coverage:
    - construction is cheap (no I/O, no loop binding);
    - ``pool`` access before startup raises ``RuntimeError``;
    - ``startup`` is idempotent under sequential and concurrent calls;
    - failed startup leaves the instance unusable (``is_started==False``);
    - ``shutdown`` is idempotent and safe against partial init;
    - ``startup`` + ``shutdown`` racing does not leave a dangling pool.

The tests use ``fakeredis.aioredis`` via a monkeypatched
``Redis.from_url`` so we exercise the real :class:`RedisConnection`
codepath without an actual Redis server.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import fakeredis.aioredis
import pytest

from gigaevo.dataplane.connection import RedisConnection
from gigaevo.dataplane.errors import StartupError


@pytest.fixture
def fake_from_url() -> Iterator[None]:
    """Replace ``aioredis.Redis.from_url`` with a fakeredis-backed builder.

    Each call gets its own :class:`fakeredis.FakeServer` so tests don't
    leak state via the shared in-memory store.
    """

    def builder(_url: str, **_kwargs: Any) -> fakeredis.aioredis.FakeRedis:
        server = fakeredis.FakeServer()
        return fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)

    with patch(
        "gigaevo.dataplane.connection.aioredis.Redis.from_url",
        side_effect=builder,
    ):
        yield


@pytest.fixture
def flaky_from_url() -> Iterator[None]:
    """``from_url`` builds a client whose first PING raises ConnectionError."""

    class _BoomRedis:
        async def ping(self) -> bool:
            raise ConnectionError("simulated boom")

        async def aclose(self) -> None:
            return None

    def builder(_url: str, **_kwargs: Any) -> _BoomRedis:
        return _BoomRedis()

    with patch(
        "gigaevo.dataplane.connection.aioredis.Redis.from_url",
        side_effect=builder,
    ):
        yield


class TestConstruction:
    def test_cheap_no_io(self) -> None:
        conn = RedisConnection("redis://x", key_prefix="p")
        assert conn.key_prefix == "p"
        assert not conn.is_started

    def test_pool_before_startup_raises(self) -> None:
        conn = RedisConnection("redis://x", key_prefix="p")
        with pytest.raises(RuntimeError, match="before startup"):
            _ = conn.pool


class TestStartup:
    async def test_startup_sets_pool(self, fake_from_url: None) -> None:
        conn = RedisConnection("redis://fake", key_prefix="p")
        await conn.startup()
        assert conn.is_started
        assert conn.pool is not None
        await conn.shutdown()

    async def test_startup_idempotent_sequential(self, fake_from_url: None) -> None:
        conn = RedisConnection("redis://fake", key_prefix="p")
        await conn.startup()
        first_pool = conn.pool
        await conn.startup()
        # Same pool instance after a second startup — no rebuild.
        assert conn.pool is first_pool
        await conn.shutdown()

    async def test_startup_concurrent_builds_one_pool(
        self, fake_from_url: None
    ) -> None:
        """Twenty parallel startup callers must observe a single pool.

        The fast-path check is not lock-protected against itself, but
        the slow path under the lifecycle lock double-checks ``_pool``
        before building. The post-condition: exactly one pool ends up
        wired into ``self._pool``.
        """
        conn = RedisConnection("redis://fake", key_prefix="p")
        await asyncio.gather(*[conn.startup() for _ in range(20)])
        assert conn.is_started
        pool = conn.pool
        # All callers must end up with the same wired pool.
        for _ in range(5):
            await conn.startup()
            assert conn.pool is pool
        await conn.shutdown()

    async def test_failed_startup_leaves_instance_unstarted(
        self, flaky_from_url: None
    ) -> None:
        conn = RedisConnection("redis://fake", key_prefix="p")
        with pytest.raises(StartupError, match="simulated boom"):
            await conn.startup()
        assert not conn.is_started
        with pytest.raises(RuntimeError):
            _ = conn.pool

    async def test_failed_startup_can_be_retried(self) -> None:
        """A failed startup does not poison the lock for retries."""
        # First call fails because from_url raises; second succeeds.
        attempts = {"n": 0}

        def builder(_url: str, **_kwargs: Any) -> fakeredis.aioredis.FakeRedis:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionError("first attempt fails")
            return fakeredis.aioredis.FakeRedis(decode_responses=True)

        with patch(
            "gigaevo.dataplane.connection.aioredis.Redis.from_url",
            side_effect=builder,
        ):
            conn = RedisConnection("redis://fake", key_prefix="p")
            with pytest.raises(StartupError):
                await conn.startup()
            assert not conn.is_started
            await conn.startup()
            assert conn.is_started
            await conn.shutdown()

    async def test_cancellation_during_ping_closes_partial_pool(self) -> None:
        """If startup is cancelled mid-PING the half-built pool is closed.

        ``asyncio.CancelledError`` is a ``BaseException``, not an
        ``Exception``, so a generic ``except Exception`` would let the
        pool leak. This test pins the explicit handling.
        """
        closed: dict[str, bool] = {"aclose_called": False}

        class _SlowRedis:
            async def ping(self) -> bool:
                # Park forever; the caller cancels.
                await asyncio.Event().wait()
                return True

            async def aclose(self) -> None:
                closed["aclose_called"] = True

        def builder(_url: str, **_kwargs: Any) -> _SlowRedis:
            return _SlowRedis()

        with patch(
            "gigaevo.dataplane.connection.aioredis.Redis.from_url",
            side_effect=builder,
        ):
            conn = RedisConnection("redis://fake", key_prefix="p")
            task = asyncio.create_task(conn.startup())
            # Let startup get to the await on ping.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not conn.is_started
            assert closed["aclose_called"], (
                "aclose was not called — the half-built pool leaked."
            )


class TestShutdown:
    async def test_shutdown_idempotent(self, fake_from_url: None) -> None:
        conn = RedisConnection("redis://fake", key_prefix="p")
        await conn.startup()
        await conn.shutdown()
        await conn.shutdown()  # second is a no-op
        assert not conn.is_started

    async def test_shutdown_before_startup_is_noop(self) -> None:
        conn = RedisConnection("redis://fake", key_prefix="p")
        await conn.shutdown()
        assert not conn.is_started

    async def test_concurrent_shutdowns(self, fake_from_url: None) -> None:
        conn = RedisConnection("redis://fake", key_prefix="p")
        await conn.startup()
        await asyncio.gather(*[conn.shutdown() for _ in range(10)])
        assert not conn.is_started


class TestStartupShutdownInterleaving:
    async def test_startup_then_shutdown_then_startup(
        self, fake_from_url: None
    ) -> None:
        conn = RedisConnection("redis://fake", key_prefix="p")
        await conn.startup()
        await conn.shutdown()
        await conn.startup()
        assert conn.is_started
        await conn.shutdown()

    async def test_interleaved_startup_shutdown_storm(
        self, fake_from_url: None
    ) -> None:
        """Twenty pairs of (startup, shutdown) racing.

        The post-condition is binary: either started with a live pool
        or fully shut down. No half-open dangling pool.
        """
        conn = RedisConnection("redis://fake", key_prefix="p")
        tasks: list[asyncio.Task[None]] = []
        for _ in range(20):
            tasks.append(asyncio.create_task(conn.startup()))
            tasks.append(asyncio.create_task(conn.shutdown()))
        await asyncio.gather(*tasks)
        # Final state may be either started or stopped — we just need
        # internal consistency: if is_started, pool is reachable; else
        # pool access raises.
        if conn.is_started:
            assert conn.pool is not None
            await conn.shutdown()
        else:
            with pytest.raises(RuntimeError):
                _ = conn.pool
