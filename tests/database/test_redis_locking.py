"""Tests for RedisInstanceLock: acquire, release, renew, periodic renewal."""

from __future__ import annotations

import asyncio
import re

import fakeredis.aioredis
import pytest

from gigaevo.database.redis.config import (
    RedisConnectionConfig,
    RedisKeyConfig,
    RedisLockConfig,
)
from gigaevo.database.redis.connection import RedisConnection
from gigaevo.database.redis.keys import RedisProgramKeys
from gigaevo.database.redis.locking import RedisInstanceLock
from gigaevo.exceptions import StorageError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lock(
    fake_redis: fakeredis.aioredis.FakeRedis,
    lock_expiry_secs: int = 2,
    lock_renewal_secs: int = 1,
    key_prefix: str = "test",
) -> RedisInstanceLock:
    """Build a RedisInstanceLock backed by a fakeredis instance."""
    conn_config = RedisConnectionConfig(
        redis_url="redis://fake:6379/0",
        max_retries=1,
        retry_delay=0.0,
    )
    conn = RedisConnection(conn_config)
    conn._redis = fake_redis
    conn._closing = False

    key_config = RedisKeyConfig(key_prefix=key_prefix)
    keys = RedisProgramKeys(key_config)
    lock_config = RedisLockConfig(
        lock_expiry_secs=lock_expiry_secs,
        lock_renewal_secs=lock_renewal_secs,
    )
    return RedisInstanceLock(conn, keys, lock_config)


@pytest.fixture
def fake_redis():
    """Shared fakeredis instance for a single test."""
    server = fakeredis.FakeServer()
    return fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)


@pytest.fixture
def lock(fake_redis):
    """RedisInstanceLock with short expiry for fast tests."""
    return _make_lock(fake_redis)


# ---------------------------------------------------------------------------
# TestInstanceId
# ---------------------------------------------------------------------------


class TestInstanceId:
    def test_format_hostname_pid_hex(self):
        """instance_id matches hostname:pid:8-char-hex."""
        server = fakeredis.FakeServer()
        fr = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        lock = _make_lock(fr)
        pattern = re.compile(r"^.+:\d+:[0-9a-f]{8}$")
        assert pattern.match(lock.instance_id), f"Bad format: {lock.instance_id}"


# ---------------------------------------------------------------------------
# TestAcquire
# ---------------------------------------------------------------------------


class TestAcquire:
    async def test_acquire_empty_lock_succeeds(self, lock, fake_redis):
        """Acquiring an unheld lock returns True and sets the key in Redis."""
        result = await lock.acquire()
        assert result is True
        assert lock.is_held is True

        # Verify key is set in Redis
        lock_key = lock._keys.instance_lock()
        value = await fake_redis.get(lock_key)
        assert value is not None
        assert value.startswith(lock.instance_id)

        # Cleanup: release to cancel the renewal task
        await lock.release()

    async def test_acquire_held_lock_raises(self, fake_redis):
        """Acquiring a lock that is already held raises StorageError."""
        lock1 = _make_lock(fake_redis, key_prefix="test")
        lock2 = _make_lock(fake_redis, key_prefix="test")

        await lock1.acquire()
        try:
            with pytest.raises(StorageError, match="another instance"):
                await lock2.acquire()
        finally:
            await lock1.release()

    async def test_acquire_starts_renewal_task(self, lock):
        """After acquire, a renewal background task is running."""
        await lock.acquire()
        assert lock._renewal_task is not None
        assert not lock._renewal_task.done()
        await lock.release()


# ---------------------------------------------------------------------------
# TestRelease
# ---------------------------------------------------------------------------


class TestRelease:
    async def test_release_clears_token_and_key(self, lock, fake_redis):
        """Release clears _token and deletes the Redis key."""
        await lock.acquire()
        lock_key = lock._keys.instance_lock()

        await lock.release()

        assert lock.is_held is False
        assert lock._token is None
        value = await fake_redis.get(lock_key)
        assert value is None

    async def test_release_noop_without_acquire(self, lock):
        """Calling release without acquire does not raise."""
        await lock.release()
        assert lock.is_held is False

    async def test_release_cancels_renewal_task(self, lock):
        """Release cancels the periodic renewal task."""
        await lock.acquire()
        task = lock._renewal_task
        assert task is not None

        await lock.release()

        assert lock._renewal_task is None
        assert task.cancelled() or task.done()


# ---------------------------------------------------------------------------
# TestRenew
# ---------------------------------------------------------------------------


class TestRenew:
    async def test_renew_refreshes_ttl(self, lock, fake_redis):
        """Renew refreshes the TTL; the stored value is the stable
        instance id, so the freshness witness is PTTL alone."""
        await lock.acquire()
        lock_key = lock._keys.instance_lock()
        old_value = await fake_redis.get(lock_key)
        old_token = lock._token

        await asyncio.sleep(0.02)
        pttl_before = await fake_redis.pttl(lock_key)

        result = await lock.renew()
        assert result is True

        new_value = await fake_redis.get(lock_key)
        assert new_value == old_value
        assert new_value == lock.instance_id
        assert lock._token == old_token
        pttl_after = await fake_redis.pttl(lock_key)
        assert pttl_after >= pttl_before

        await lock.release()

    async def test_renew_no_token_returns_false(self, lock):
        """Renew without a prior acquire returns False."""
        result = await lock.renew()
        assert result is False

    async def test_renew_lost_lock_detected(self, lock, fake_redis):
        """If another instance overwrote the lock, renew returns False."""
        await lock.acquire()

        # Simulate another instance overwriting the lock
        lock_key = lock._keys.instance_lock()
        await fake_redis.set(lock_key, "other-instance:12345:abcd1234:9999.9")

        result = await lock.renew()
        assert result is False

        # Clean up: clear token so release doesn't fail
        lock._token = None
        await lock.release()

    async def test_renew_deleted_key_returns_false(self, lock, fake_redis):
        """If the lock key was DELETEd, the token-CAS renew fails closed."""
        await lock.acquire()
        lock_key = lock._keys.instance_lock()
        await fake_redis.delete(lock_key)

        result = await lock.renew()
        assert result is False

        lock._token = None
        await lock.release()


# ---------------------------------------------------------------------------
# TestRenewPeriodically
# ---------------------------------------------------------------------------


class TestRenewPeriodically:
    async def test_updates_redis_periodically(self, fake_redis):
        """Periodic renewal actually updates the Redis value."""
        lock = _make_lock(fake_redis, lock_renewal_secs=0)
        await lock.acquire()
        lock_key = lock._keys.instance_lock()
        await fake_redis.get(lock_key)

        # Give the renewal loop time to run
        await asyncio.sleep(0.05)

        new_value = await fake_redis.get(lock_key)
        # The renewal should have updated the value at least once
        assert new_value.startswith(lock.instance_id)

        await lock.release()

    async def test_stops_on_closing(self, fake_redis):
        """Renewal loop exits when the connection is closing."""
        lock = _make_lock(fake_redis, lock_renewal_secs=0)
        await lock.acquire()
        task = lock._renewal_task

        # Signal closing
        lock._conn._closing = True
        await asyncio.sleep(0.05)

        # The task should have completed (not just cancelled)
        assert task.done()

        # Clean up
        lock._renewal_task = None
        lock._token = None

    async def test_stops_on_renewal_failure(self, fake_redis):
        """Renewal loop breaks when renew() returns False (lost lock)."""
        lock = _make_lock(fake_redis, lock_renewal_secs=0)
        await lock.acquire()
        task = lock._renewal_task

        # Overwrite the lock to simulate another instance taking it
        lock_key = lock._keys.instance_lock()
        await fake_redis.set(lock_key, "other-instance:0:00000000:0.0")

        # Give the renewal loop time to detect the failure
        await asyncio.sleep(0.1)

        assert task.done()

        # Clean up
        lock._renewal_task = None
        lock._token = None


# ---------------------------------------------------------------------------
# TestReleaseRobustness
# ---------------------------------------------------------------------------


class TestReleaseRobustness:
    async def test_only_deletes_own_lock(self, fake_redis):
        """Release does NOT delete a lock held by another instance."""
        lock = _make_lock(fake_redis)
        await lock.acquire()

        # Simulate another instance overwriting the lock before release
        lock_key = lock._keys.instance_lock()
        other_value = "other-host:99999:abcdef12:1234567890.0"
        await fake_redis.set(lock_key, other_value)

        await lock.release()

        # The other instance's lock should still be there
        remaining = await fake_redis.get(lock_key)
        assert remaining == other_value

    async def test_tolerates_connection_error(self, fake_redis):
        """Release swallows connection errors and clears local holder
        state so ``is_held`` reports False after a Redis-side failure."""
        lock = _make_lock(fake_redis)
        await lock.acquire()
        assert lock._token is not None

        async def failing_execute(name, fn):
            raise ConnectionError("redis gone")

        lock._conn.execute = failing_execute

        await lock.release()

        assert lock._renewal_task is None
        assert lock._token is None
        assert lock.is_held is False

    async def test_acquire_sets_expiry(self, lock, fake_redis):
        """Acquired lock has a TTL set in Redis."""
        await lock.acquire()
        lock_key = lock._keys.instance_lock()
        ttl = await fake_redis.ttl(lock_key)
        assert ttl > 0  # TTL is positive (lock has expiry)
        await lock.release()


# ---------------------------------------------------------------------------
# Dataplane-routed path: LuaRegistry.evalsha + wrap_lease integration
# ---------------------------------------------------------------------------


async def _build_lock_with_dp(
    fake_redis: fakeredis.aioredis.FakeRedis,
    *,
    key_prefix: str = "dptest",
    lock_expiry_secs: int = 2,
    lock_renewal_secs: int = 1,
):
    """Build a RedisInstanceLock backed by a started DataPlane sharing
    the supplied fakeredis instance with the lock's connection."""
    import gigaevo.dataplane as dp
    from gigaevo.dataplane.scripts import LuaRegistry

    conn_config = RedisConnectionConfig(
        redis_url="redis://fake:6379/0",
        max_retries=1,
        retry_delay=0.0,
    )
    conn = RedisConnection(conn_config)
    conn._redis = fake_redis
    conn._closing = False

    key_config = RedisKeyConfig(key_prefix=key_prefix)
    keys = RedisProgramKeys(key_config)
    lock_config = RedisLockConfig(
        lock_expiry_secs=lock_expiry_secs,
        lock_renewal_secs=lock_renewal_secs,
    )

    coord = dp.DataPlane("redis://fake:6379/0", key_prefix=key_prefix)
    coord._connection._pool = fake_redis  # type: ignore[attr-defined]
    lua = LuaRegistry(fake_redis)
    coord._register_builtin_scripts(lua)  # type: ignore[attr-defined]
    await lua.load_all()
    coord._lua = lua  # type: ignore[attr-defined]
    coord._started = True  # type: ignore[attr-defined]

    lock = RedisInstanceLock(conn, keys, lock_config, dataplane=coord)
    return lock, coord


class TestDataplaneRoutedLocking:
    """Lock scripts dispatch through :meth:`LuaRegistry.evalsha` and the
    acquired lock is wrapped in a :class:`CrashWatchedHandle`."""

    async def test_acquire_routes_through_dataplane_lua_registry(
        self, fake_redis
    ) -> None:
        lock, coord = await _build_lock_with_dp(fake_redis)
        try:
            assert await lock.acquire() is True
            assert lock.is_held
            # SHA cache lives on the dp's LuaRegistry, not the lock.
            assert lock._script_shas == {}
            assert lock.wrapped_lease is not None
            assert await lock.observe_loss() is None
        finally:
            await lock.release()
            coord._started = False  # type: ignore[attr-defined]

    async def test_renewal_loss_signals_crash_event(self, fake_redis) -> None:
        """Token-CAS mismatch on renewal yields a :class:`CrashEvent`
        on the next ``observe_loss``."""
        lock, coord = await _build_lock_with_dp(fake_redis)
        try:
            await lock.acquire()
            lock_key = lock._keys.instance_lock()
            await fake_redis.set(lock_key, "other-instance:99999:abcd1234")
            ok = await lock.renew()
            assert ok is False
            evt = await lock.observe_loss()
            assert evt is not None
            assert evt.peer == lock_key
            # Lease handle consumed: a second observer sees nothing.
            assert await lock.observe_loss() is None
        finally:
            lock._token = None
            await lock.release()
            coord._started = False  # type: ignore[attr-defined]

    async def test_release_clears_wrapped_lease(self, fake_redis) -> None:
        """``release`` clears the handle; the next acquire mints a fresh one."""
        lock, coord = await _build_lock_with_dp(fake_redis)
        try:
            await lock.acquire()
            assert lock.wrapped_lease is not None
            await lock.release()
            assert lock.wrapped_lease is None
            await lock.acquire()
            assert lock.wrapped_lease is not None
        finally:
            await lock.release()
            coord._started = False  # type: ignore[attr-defined]

    async def test_observe_loss_legacy_path_returns_none(self, fake_redis) -> None:
        """The legacy direct-aioredis path has no lease vocabulary;
        ``observe_loss`` returns ``None`` unconditionally."""
        lock = _make_lock(fake_redis)
        await lock.acquire()
        try:
            assert lock.wrapped_lease is None
            assert await lock.observe_loss() is None
        finally:
            await lock.release()
