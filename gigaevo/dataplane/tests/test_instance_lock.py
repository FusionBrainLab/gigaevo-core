"""Integration tests for the instance-lock API.

Covers the three Lua scripts (acquire / renew / release) and the
matching :class:`DataPlane` methods, including the background renewal
task and the :class:`OneShotFlag` signal on detected loss.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest

import gigaevo.dataplane as dp


@pytest.fixture
async def coord() -> AsyncIterator[dp.DataPlane]:
    """Coordinator wired to an isolated fakeredis instance."""
    server = fakeredis.FakeServer()
    coord = dp.DataPlane("redis://embedded/0", key_prefix="test")
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    coord._connection._pool = fake  # type: ignore[attr-defined]
    from gigaevo.dataplane.scripts import LuaRegistry

    lua = LuaRegistry(fake)
    coord._register_builtin_scripts(lua)  # type: ignore[attr-defined]
    await lua.load_all()
    coord._lua = lua  # type: ignore[attr-defined]
    coord._started = True  # type: ignore[attr-defined]
    try:
        yield coord
    finally:
        await coord._cancel_renewers()  # type: ignore[attr-defined]
        coord._started = False  # type: ignore[attr-defined]
        coord._lua = None  # type: ignore[attr-defined]
        coord._connection._pool = None  # type: ignore[attr-defined]
        await fake.aclose()  # type: ignore[attr-defined]


# ── acquire ──────────────────────────────────────────────────────────


class TestAcquire:
    async def test_uncontended_returns_ok(self, coord: dp.DataPlane) -> None:
        result = await coord.acquire_instance_lock(dp.KeyPrefix("run-1"), ttl_s=30.0)
        assert isinstance(result, dp.Ok)
        lease = result.value
        assert lease.key == "test:lock:run-1"
        assert lease.ttl_s == 30.0
        assert not lease.flag.is_set()
        assert isinstance(lease.token, str) and len(lease.token) > 0

    async def test_contended_returns_lockheld(self, coord: dp.DataPlane) -> None:
        first = await coord.acquire_instance_lock(dp.KeyPrefix("run-1"), ttl_s=30.0)
        assert isinstance(first, dp.Ok)
        second = await coord.acquire_instance_lock(dp.KeyPrefix("run-1"), ttl_s=30.0)
        assert isinstance(second, dp.Err)
        assert isinstance(second.error, dp.LockHeld)
        assert second.error.key == "test:lock:run-1"
        # Holder is set to the first lease's token (best-effort holder
        # field; redacted in str(err) but accessible programmatically).
        assert second.error.holder == first.value.token

    async def test_different_prefixes_acquire_independently(
        self, coord: dp.DataPlane
    ) -> None:
        a = await coord.acquire_instance_lock(dp.KeyPrefix("a"), ttl_s=30.0)
        b = await coord.acquire_instance_lock(dp.KeyPrefix("b"), ttl_s=30.0)
        assert isinstance(a, dp.Ok)
        assert isinstance(b, dp.Ok)
        assert a.value.token != b.value.token

    async def test_unique_tokens_per_acquisition(self, coord: dp.DataPlane) -> None:
        a = await coord.acquire_instance_lock(dp.KeyPrefix("a"), ttl_s=30.0)
        assert isinstance(a, dp.Ok)
        await coord.release_instance_lock(a.value)
        b = await coord.acquire_instance_lock(dp.KeyPrefix("a"), ttl_s=30.0)
        assert isinstance(b, dp.Ok)
        assert a.value.token != b.value.token


# ── renew ────────────────────────────────────────────────────────────


class TestRenew:
    async def test_renew_with_correct_token_succeeds(self, coord: dp.DataPlane) -> None:
        acquired = await coord.acquire_instance_lock(
            dp.KeyPrefix("renewable"), ttl_s=30.0
        )
        assert isinstance(acquired, dp.Ok)
        renewed = await coord.renew_instance_lock(acquired.value, ttl_s=60.0)
        assert isinstance(renewed, dp.Ok)
        assert renewed.value.token == acquired.value.token
        assert renewed.value.ttl_s == 60.0
        # The flag is preserved across renewals.
        assert renewed.value.flag is acquired.value.flag

    async def test_renew_with_wrong_token_returns_locklost(
        self, coord: dp.DataPlane
    ) -> None:
        acquired = await coord.acquire_instance_lock(
            dp.KeyPrefix("foreign"), ttl_s=30.0
        )
        assert isinstance(acquired, dp.Ok)
        # Construct a fake lease pointing at the same key but a different token.
        fake_lease = dp.InstanceLease(
            token=dp.LeaseToken("not-the-real-token"),
            key=acquired.value.key,
            ttl_s=30.0,
            expires_at_monotonic=acquired.value.expires_at_monotonic,
            flag=dp.OneShotFlag(),
        )
        renewed = await coord.renew_instance_lock(fake_lease, ttl_s=30.0)
        assert isinstance(renewed, dp.Err)
        assert isinstance(renewed.error, dp.LockLost)
        assert renewed.error.key == acquired.value.key

    async def test_renew_after_external_delete_returns_locklost(
        self, coord: dp.DataPlane
    ) -> None:
        acquired = await coord.acquire_instance_lock(dp.KeyPrefix("del-me"), ttl_s=30.0)
        assert isinstance(acquired, dp.Ok)
        # Cancel the renewer to keep the test deterministic.
        await coord._cancel_renewers()  # type: ignore[attr-defined]
        # Simulate TTL expiry or an external DEL.
        await coord._connection.pool.delete(acquired.value.key)  # type: ignore[misc]
        renewed = await coord.renew_instance_lock(acquired.value, ttl_s=30.0)
        assert isinstance(renewed, dp.Err)
        assert isinstance(renewed.error, dp.LockLost)


# ── release ──────────────────────────────────────────────────────────


class TestAcquireInputValidation:
    """Server-side defence for malformed inputs.

    The Python wrapper rejects non-positive ``ttl_s`` before issuing
    EVALSHA — covered by the existing wrapper-level check. These tests
    target the Lua script directly to pin its behaviour as the
    last-line-of-defence belt under the wrapper's suspenders.
    """

    async def test_direct_zero_ttl_call_errors(self, coord: dp.DataPlane) -> None:
        lua = coord._lua  # type: ignore[attr-defined]
        with pytest.raises(Exception):  # noqa: PT011,BLE001
            await lua.evalsha(
                "instance_lock_acquire",
                keys=["test:lock:direct-zero"],
                args=["sometoken", 0],
            )

    async def test_direct_negative_ttl_call_errors(self, coord: dp.DataPlane) -> None:
        lua = coord._lua  # type: ignore[attr-defined]
        with pytest.raises(Exception):  # noqa: PT011,BLE001
            await lua.evalsha(
                "instance_lock_acquire",
                keys=["test:lock:direct-neg"],
                args=["sometoken", -1],
            )

    async def test_direct_empty_token_errors(self, coord: dp.DataPlane) -> None:
        lua = coord._lua  # type: ignore[attr-defined]
        with pytest.raises(Exception):  # noqa: PT011,BLE001
            await lua.evalsha(
                "instance_lock_acquire",
                keys=["test:lock:direct-empty"],
                args=["", 5000],
            )


class TestRelease:
    async def test_release_drops_the_key(self, coord: dp.DataPlane) -> None:
        acquired = await coord.acquire_instance_lock(
            dp.KeyPrefix("releaseable"), ttl_s=30.0
        )
        assert isinstance(acquired, dp.Ok)
        await coord.release_instance_lock(acquired.value)
        # Verify the key is gone.
        value = await coord._connection.pool.get(acquired.value.key)  # type: ignore[misc]
        assert value is None
        # A fresh acquire must succeed.
        second = await coord.acquire_instance_lock(
            dp.KeyPrefix("releaseable"), ttl_s=30.0
        )
        assert isinstance(second, dp.Ok)

    async def test_release_with_wrong_token_is_no_op(self, coord: dp.DataPlane) -> None:
        acquired = await coord.acquire_instance_lock(
            dp.KeyPrefix("guarded"), ttl_s=30.0
        )
        assert isinstance(acquired, dp.Ok)
        await coord._cancel_renewers()  # type: ignore[attr-defined]
        fake_lease = dp.InstanceLease(
            token=dp.LeaseToken("attacker-token"),
            key=acquired.value.key,
            ttl_s=30.0,
            expires_at_monotonic=acquired.value.expires_at_monotonic,
            flag=dp.OneShotFlag(),
        )
        await coord.release_instance_lock(fake_lease)
        # Real lock is still held — verify with a contention probe.
        contended = await coord.acquire_instance_lock(
            dp.KeyPrefix("guarded"), ttl_s=30.0
        )
        assert isinstance(contended, dp.Err)
        assert isinstance(contended.error, dp.LockHeld)

    async def test_release_idempotent(self, coord: dp.DataPlane) -> None:
        acquired = await coord.acquire_instance_lock(
            dp.KeyPrefix("idempotent"), ttl_s=30.0
        )
        assert isinstance(acquired, dp.Ok)
        await coord.release_instance_lock(acquired.value)
        # Second call must not raise — release-on-released is a no-op.
        await coord.release_instance_lock(acquired.value)


# ── auto-renewal ─────────────────────────────────────────────────────


class TestAutoRenewal:
    async def test_renewer_keeps_lock_alive_across_intervals(
        self, coord: dp.DataPlane
    ) -> None:
        """A short-TTL lease must survive multiple natural-expiry windows.

        With ttl=1s and renew-ratio=3, the renewer fires every ~0.33s.
        Wait 1.2s (longer than ttl) — the lock must still be ours,
        which we verify by observing that a contention probe still sees
        the same holder token.
        """
        acquired = await coord.acquire_instance_lock(
            dp.KeyPrefix("autoreneweable"), ttl_s=1.0
        )
        assert isinstance(acquired, dp.Ok)
        await asyncio.sleep(1.2)
        # Probe via the GET-on-failure path of acquire.
        contended = await coord.acquire_instance_lock(
            dp.KeyPrefix("autoreneweable"), ttl_s=1.0
        )
        assert isinstance(contended, dp.Err)
        assert isinstance(contended.error, dp.LockHeld)
        assert contended.error.holder == acquired.value.token
        # Flag must still be unset — the lease is healthy.
        assert not acquired.value.flag.is_set()

    async def test_flag_fires_on_external_loss(self, coord: dp.DataPlane) -> None:
        """An external DEL between renewals must surface as OneShotFlag.

        With ttl=0.3s the renewer wakes every ~0.1s. After acquire we
        wait one renewal cycle for the loop to enter sleep, then DEL
        the key directly; the next renewal observes LockLost and
        signals the flag.
        """
        acquired = await coord.acquire_instance_lock(
            dp.KeyPrefix("loseable"), ttl_s=0.3
        )
        assert isinstance(acquired, dp.Ok)
        await asyncio.sleep(0.05)  # let renewer enter sleep
        await coord._connection.pool.delete(acquired.value.key)  # type: ignore[misc]
        # Wait long enough for the renewer to wake, renew (fail), and signal.
        for _ in range(20):
            if acquired.value.flag.is_set():
                break
            await asyncio.sleep(0.05)
        assert acquired.value.flag.is_set()

    async def test_release_cancels_renewer(self, coord: dp.DataPlane) -> None:
        acquired = await coord.acquire_instance_lock(
            dp.KeyPrefix("cancel-test"), ttl_s=30.0
        )
        assert isinstance(acquired, dp.Ok)
        # Renewer task is tracked on the coordinator.
        renewers: dict[dp.LeaseToken, asyncio.Task[None]] = coord._renewers  # type: ignore[attr-defined]
        assert acquired.value.token in renewers
        task = renewers[acquired.value.token]
        await coord.release_instance_lock(acquired.value)
        # Token is dropped from the renewer table.
        assert acquired.value.token not in renewers
        # Task is cancelled.
        await asyncio.sleep(0)  # yield once so cancellation propagates
        assert task.cancelled() or task.done()

    async def test_cancel_renewers_drains_all(self, coord: dp.DataPlane) -> None:
        a = await coord.acquire_instance_lock(dp.KeyPrefix("a"), ttl_s=30.0)
        b = await coord.acquire_instance_lock(dp.KeyPrefix("b"), ttl_s=30.0)
        c = await coord.acquire_instance_lock(dp.KeyPrefix("c"), ttl_s=30.0)
        assert isinstance(a, dp.Ok)
        assert isinstance(b, dp.Ok)
        assert isinstance(c, dp.Ok)
        renewers: dict[dp.LeaseToken, asyncio.Task[None]] = coord._renewers  # type: ignore[attr-defined]
        assert len(renewers) == 3
        await coord._cancel_renewers()  # type: ignore[attr-defined]
        assert len(renewers) == 0

    async def test_release_waits_for_renewer_exit(self, coord: dp.DataPlane) -> None:
        """``release_instance_lock`` must not return until the renewer is gone.

        Without the explicit ``await`` on cancellation the renewer could
        still be mid-EXPIRE when the release path issues its DEL,
        leaving the lock un-bounded for one renew interval. The fix
        gathers on the task before issuing the release.
        """
        acquired = await coord.acquire_instance_lock(
            dp.KeyPrefix("await-cancel"), ttl_s=30.0
        )
        assert isinstance(acquired, dp.Ok)
        renewers: dict[dp.LeaseToken, asyncio.Task[None]] = coord._renewers  # type: ignore[attr-defined]
        task = renewers[acquired.value.token]
        await coord.release_instance_lock(acquired.value)
        # The task is either cancelled or done — never still running.
        assert task.done() or task.cancelled()

    async def test_renewer_exception_signals_flag(self, coord: dp.DataPlane) -> None:
        """An exception inside the renewer must escalate to the OneShotFlag.

        We synthesise the failure by closing the underlying pool so the
        next renewal raises a non-Err exception. The watchdog must
        translate that into a flag signal so the holder observes the
        loss on its next dataplane call instead of silently leaking.
        """
        acquired = await coord.acquire_instance_lock(
            dp.KeyPrefix("renew-explode"), ttl_s=0.3
        )
        assert isinstance(acquired, dp.Ok)
        # Drop the registry so the next ``renew_instance_lock`` raises
        # ``NotStartedError`` from within the renewer loop.
        coord._lua = None  # type: ignore[attr-defined]
        for _ in range(30):
            if acquired.value.flag.is_set():
                break
            await asyncio.sleep(0.05)
        assert acquired.value.flag.is_set()
