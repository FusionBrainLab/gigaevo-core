"""E2E tests for steady-state adversarial co-evolution sync.

Tests dual-population ProgressBasedSyncHook interaction using fakeredis.
Verifies that two populations can advance with mutual sync constraints
without deadlock.
"""

from __future__ import annotations

import asyncio

import fakeredis.aioredis

from gigaevo.adversarial.sync import ProgressBasedSyncHook


def _make_hook(
    server: fakeredis.FakeServer,
    *,
    own_db: int,
    opponent_db: int,
    prefix: str = "test",
    min_delta: int = 5,
    sync_every_n_epochs: int = 1,
) -> ProgressBasedSyncHook:
    """Create a ProgressBasedSyncHook backed by a fakeredis server."""
    hook = ProgressBasedSyncHook(
        host="localhost",
        port=6379,
        sources=[{"db": opponent_db, "prefix": prefix}],
        min_delta=min_delta,
        sync_every_n_epochs=sync_every_n_epochs,
        timeout=5.0,
        poll_interval=0.01,
    )
    # Wire up fakeredis clients for both DBs
    for db in (own_db, opponent_db):
        hook._redis_clients[db] = fakeredis.aioredis.FakeRedis(
            server=server, db=db, decode_responses=True
        )
    return hook


async def _set_progress(
    server: fakeredis.FakeServer, db: int, prefix: str, value: int
) -> None:
    """Set engine:programs_processed in fakeredis."""
    r = fakeredis.aioredis.FakeRedis(server=server, db=db, decode_responses=True)
    await r.hset(f"{prefix}:run_state", "engine:programs_processed", str(value))
    await r.aclose()


class TestDualPopulationSync:
    async def test_two_populations_advance_with_sync(self) -> None:
        """Two hooks pointing at each other's DBs both advance without deadlock.

        Simulates alternating advancement:
        1. Pop A advances its progress → Pop B's hook unblocks
        2. Pop B advances its progress → Pop A's hook unblocks
        """
        server = fakeredis.FakeServer()
        prefix = "test"

        hook_a = _make_hook(server, own_db=1, opponent_db=2, prefix=prefix, min_delta=5)
        hook_b = _make_hook(server, own_db=2, opponent_db=1, prefix=prefix, min_delta=5)

        # Initialize: both start at 0
        await _set_progress(server, 1, prefix, 0)
        await _set_progress(server, 2, prefix, 0)

        # First call: both record baseline (no blocking)
        await hook_a()
        await hook_b()
        assert hook_a._last_progress == 0
        assert hook_b._last_progress == 0

        # Pop A advances its own progress to 10
        await _set_progress(server, 1, prefix, 10)

        # Pop B's hook should now unblock (opponent=DB1, progress=10 >= 0+5)
        await hook_b()
        assert hook_b._last_progress == 10

        # Pop B advances its own progress to 8
        await _set_progress(server, 2, prefix, 8)

        # Pop A's hook should now unblock (opponent=DB2, progress=8 >= 0+5)
        await hook_a()
        assert hook_a._last_progress == 8

    async def test_asymmetric_k1_ratio(self) -> None:
        """Pop B with sync_every_n_epochs=3 runs 3 epochs per sync.

        Pop A syncs every epoch; Pop B syncs every 3rd epoch.
        After 3 calls to each, Pop A should have synced 3 times
        and Pop B should have synced 1 time.
        """
        server = fakeredis.FakeServer()
        prefix = "test"

        hook_a = _make_hook(
            server,
            own_db=1,
            opponent_db=2,
            prefix=prefix,
            min_delta=5,
            sync_every_n_epochs=1,
        )
        hook_b = _make_hook(
            server,
            own_db=2,
            opponent_db=1,
            prefix=prefix,
            min_delta=5,
            sync_every_n_epochs=3,
        )

        await _set_progress(server, 1, prefix, 0)
        await _set_progress(server, 2, prefix, 0)

        # Baseline calls
        await hook_a()
        await hook_b()  # baseline (skipped by K:1? No — first actual sync)
        # For hook_b with sync_every=3:
        #   call 1: epoch_count=1 < 3 → skip (no-op)
        # But baseline is recorded on first NON-skipped call.
        # Let's trace: first __call__: epoch_count=0+1=1 < 3 → skip.
        # So hook_b._last_progress is still -1 (sentinel)

        # Advance Pop A so hook_b can eventually sync
        await _set_progress(server, 1, prefix, 20)

        # Call 2 for hook_b: epoch_count=1+1=2 < 3 → skip
        await hook_b()
        assert hook_b._last_progress == -1  # still sentinel

        # Call 3 for hook_b: epoch_count=2+1=3 → sync! Records baseline.
        await hook_b()
        assert hook_b._last_progress == 20  # baseline recorded from DB1

        # Meanwhile, hook_a syncs every epoch (needs DB2 to advance)
        await _set_progress(server, 2, prefix, 10)
        await hook_a()  # baseline was 0, now 10 >= 0+5 → advance
        assert hook_a._last_progress == 10

        await _set_progress(server, 2, prefix, 20)
        await hook_a()  # 20 >= 10+5 → advance
        assert hook_a._last_progress == 20

    async def test_no_deadlock_with_concurrent_hooks(self) -> None:
        """Two hooks waiting concurrently unblock when opponent progresses.

        Simulates the real scenario: both populations call their sync hook
        at the same time. A background task advances progress for both.
        """
        server = fakeredis.FakeServer()
        prefix = "test"

        hook_a = _make_hook(server, own_db=1, opponent_db=2, prefix=prefix, min_delta=5)
        hook_b = _make_hook(server, own_db=2, opponent_db=1, prefix=prefix, min_delta=5)

        await _set_progress(server, 1, prefix, 0)
        await _set_progress(server, 2, prefix, 0)

        # Baseline calls
        await hook_a()
        await hook_b()

        # Both now need opponent to advance by 5 before they unblock.
        # Simulate a background "evolution" that advances both.
        async def advance_both():
            await asyncio.sleep(0.05)
            await _set_progress(server, 1, prefix, 10)
            await _set_progress(server, 2, prefix, 10)

        advancer = asyncio.create_task(advance_both())

        # Both hooks wait concurrently — should unblock once advancer runs
        await asyncio.wait_for(
            asyncio.gather(hook_a(), hook_b()),
            timeout=3.0,
        )

        await advancer
        assert hook_a._last_progress == 10
        assert hook_b._last_progress == 10
