"""E2E tests for steady-state adversarial co-evolution sync.

Tests dual-population ProgressBasedSyncHook interaction using fakeredis.
Verifies that two populations can advance with mutual sync constraints
without deadlock.
"""

from __future__ import annotations

import asyncio

import fakeredis.aioredis

from gigaevo.adversarial.sync import ProgressBasedSyncHook
from gigaevo.evolution.engine.snapshot import EngineSnapshot


def _make_hook(
    server: fakeredis.FakeServer,
    *,
    own_db: int,
    opponent_db: int,
    prefix: str = "test",
    drift_cap: int = 5,
    sync_every_n_epochs: int = 1,
) -> ProgressBasedSyncHook:
    """Create a ProgressBasedSyncHook backed by a fakeredis server."""
    hook = ProgressBasedSyncHook(
        host="localhost",
        port=6379,
        own_db=own_db,
        own_prefix=prefix,
        sources=[{"db": opponent_db, "prefix": prefix}],
        drift_cap=drift_cap,
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
    """Set the engine:snapshot programs_processed in fakeredis."""
    r = fakeredis.aioredis.FakeRedis(server=server, db=db, decode_responses=True)
    snap_json = EngineSnapshot(programs_processed=value).model_dump_json()
    await r.hset(f"{prefix}:run_state", "engine:snapshot", snap_json)
    await r.aclose()


class TestDualPopulationSync:
    async def test_two_populations_advance_with_sync(self) -> None:
        """Two hooks pointing at each other's DBs both advance without deadlock.

        Drift-cap semantics: only the ahead side blocks. When both at equal progress,
        neither blocks. When A leads B, A blocks until B catches up.
        """
        server = fakeredis.FakeServer()
        prefix = "test"

        hook_a = _make_hook(server, own_db=1, opponent_db=2, prefix=prefix, drift_cap=5)
        hook_b = _make_hook(server, own_db=2, opponent_db=1, prefix=prefix, drift_cap=5)

        # Initialize: both start at 0
        await _set_progress(server, 1, prefix, 0)
        await _set_progress(server, 2, prefix, 0)

        # First call: both record baseline (no blocking)
        await hook_a()
        await hook_b()

        # Pop A advances to 10, Pop B stays at 0
        # Drift = A(10) - B(0) = 10 > cap(5) → A blocks, B unblocks
        await _set_progress(server, 1, prefix, 10)
        await hook_b()  # B: opponent(A)=10, own=0, drift=10 > 5 → false, unblocks

        # Pop B advances to 8
        # Drift = A(10) - B(8) = 2 <= cap(5) → A unblocks
        await _set_progress(server, 2, prefix, 8)
        await hook_a()  # A: opponent(B)=8, own=10, drift=2 <= 5 → true, unblocks

    async def test_asymmetric_k1_ratio(self) -> None:
        """Pop B with sync_every_n_epochs=3 runs 3 epochs per sync.

        Pop A syncs every epoch; Pop B syncs every 3rd epoch.
        Verifies async hook skipping does not interfere with drift-cap logic.
        """
        server = fakeredis.FakeServer()
        prefix = "test"

        hook_a = _make_hook(
            server,
            own_db=1,
            opponent_db=2,
            prefix=prefix,
            drift_cap=5,
            sync_every_n_epochs=1,
        )
        hook_b = _make_hook(
            server,
            own_db=2,
            opponent_db=1,
            prefix=prefix,
            drift_cap=5,
            sync_every_n_epochs=3,
        )

        await _set_progress(server, 1, prefix, 0)
        await _set_progress(server, 2, prefix, 0)

        # First call: both record baseline
        await hook_a()
        # hook_b first call: epoch_count=1 < 3 → skip, no-op
        await hook_b()

        # Advance Pop A
        await _set_progress(server, 1, prefix, 20)

        # hook_b call 2: epoch_count=2 < 3 → skip
        await hook_b()

        # hook_b call 3: epoch_count=3 → sync! Check drift.
        # own(B)=0, opponent(A)=20, drift=0-20=-20 <= 5 → unblocks
        await hook_b()

        # hook_a: own(A)=20, opponent(B)=0, drift=20-0=20 > 5 → blocks
        # until B advances
        await _set_progress(server, 2, prefix, 10)
        # own(A)=20, opponent(B)=10, drift=10 > 5 → still blocks
        # own(A)=20, opponent(B)=16, drift=4 <= 5 → unblocks
        await _set_progress(server, 2, prefix, 16)
        await hook_a()

    async def test_no_deadlock_with_concurrent_hooks(self) -> None:
        """Two hooks waiting concurrently never deadlock (drift-cap property).

        Simulates real scenario: both populations call their sync hook at the
        same time with divergent progress. Drift-cap semantics ensure only the
        ahead side ever blocks, so no mutual deadlock.
        """
        server = fakeredis.FakeServer()
        prefix = "test"

        hook_a = _make_hook(server, own_db=1, opponent_db=2, prefix=prefix, drift_cap=5)
        hook_b = _make_hook(server, own_db=2, opponent_db=1, prefix=prefix, drift_cap=5)

        await _set_progress(server, 1, prefix, 0)
        await _set_progress(server, 2, prefix, 0)

        # Baseline calls
        await hook_a()
        await hook_b()

        # Set A ahead: A=10, B=0 → drift=10 > 5 → A blocks, B unblocks
        await _set_progress(server, 1, prefix, 10)
        await _set_progress(server, 2, prefix, 0)

        # Background task advances B so A can unblock
        async def advance_b():
            await asyncio.sleep(0.05)
            await _set_progress(server, 2, prefix, 6)  # drift=10-6=4 <= 5 → A unblocks

        advancer = asyncio.create_task(advance_b())

        # Both hooks called concurrently:
        # - A blocks (drift=10 > 5)
        # - B unblocks (drift=0-10=-10 <= 5)
        # As advancer runs, A unblocks → no deadlock
        await asyncio.wait_for(
            asyncio.gather(hook_a(), hook_b()),
            timeout=3.0,
        )

        await advancer
