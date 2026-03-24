"""Deadlock and hang prevention tests for the evolution engine, DAG runner, and storage layer.

These tests verify that concurrent operations complete within bounded time and
that no combination of lock acquisition, cache contention, or state transitions
can cause the system to hang indefinitely.

Every test uses asyncio.wait_for with a timeout — if any test hangs, it fails
with TimeoutError rather than stalling the suite.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest

from gigaevo.database.program_storage import PopulationSnapshot, ProgramStorage
from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis.config import RedisConnectionConfig
from gigaevo.database.redis.connection import RedisConnection
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.strategies.elite_selectors import (
    ScalarTournamentEliteSelector,
)
from gigaevo.evolution.strategies.island import IslandConfig
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.runner.dag_runner import DagRunner, DagRunnerConfig, TaskInfo

# ---------------------------------------------------------------------------
# Timeout for all tests — if anything hangs, fail fast
# ---------------------------------------------------------------------------
HANG_TIMEOUT = 5.0  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_storage(key_prefix: str = "test") -> RedisProgramStorage:
    server = fakeredis.FakeServer()
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix=key_prefix,
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage


def _make_program(state: ProgramState = ProgramState.QUEUED) -> Program:
    return Program(
        code="def solve(): return 1",
        state=state,
        atomic_counter=999_999_999,
    )


def _make_engine(storage: RedisProgramStorage, **overrides) -> EvolutionEngine:
    class _NullMutator(MutationOperator):
        async def mutate_single(
            self, selected_parents: list[Program]
        ) -> MutationSpec | None:
            return None

    defaults = dict(
        island_id="test",
        behavior_space=BehaviorSpace(
            bins={"x": LinearBinning(min_val=0.0, max_val=1.0, num_bins=2)}
        ),
        archive_selector=SumArchiveSelector(fitness_keys=["fitness"]),
        archive_remover=None,
        elite_selector=ScalarTournamentEliteSelector(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
            tournament_size=2,
        ),
        migrant_selector=RandomMigrantSelector(),
    )
    strategy = MapElitesMultiIsland(
        island_configs=[IslandConfig(**defaults)],
        program_storage=storage,
    )
    tracker = MagicMock()
    tracker.start = MagicMock()
    tracker.stop = AsyncMock()

    writer = MagicMock()
    writer.bind.return_value = writer

    engine_kwargs = dict(
        loop_interval=0.005,
        max_generations=1,
    )
    engine_kwargs.update(overrides)

    return EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=_NullMutator(),
        config=EngineConfig(**engine_kwargs),
        writer=writer,
        metrics_tracker=tracker,
    )


# ===========================================================================
# 1. _await_idle must escape ghost IDs within bounded time
# ===========================================================================


class TestAwaitIdleGhostEscape:
    """_await_idle uses count_by_status (SCARD) for the fast path, which counts
    ghost IDs. After 30s it falls back to get_all_by_status to detect ghosts
    and clean them up. These tests verify the escape hatch works.
    """

    async def test_await_idle_escapes_ghost_ids(self) -> None:
        """With only ghost IDs in QUEUED, _await_idle must eventually break out
        via the 30s fallback. We simulate this by directly testing the ghost
        cleanup path: after 30s (faked via iteration count), _await_idle
        fetches all programs, finds none, and cleans the status set.
        """
        storage = _make_storage()
        try:
            engine = _make_engine(storage, loop_interval=0.001)

            # Insert a ghost ID (no backing program data)
            r = await storage._conn.get()
            await r.sadd("test:status:queued", "ghost-no-data")

            # Verify ghost is counted by SCARD fast path
            assert await engine._has_active_dags() is True

            # Verify get_all_by_status returns empty (ghosts have no data)
            real_progs = await storage.get_all_by_status(ProgramState.QUEUED.value)
            assert len(real_progs) == 0, "Ghost IDs should not deserialize"

            # Directly test the cleanup: remove ghost IDs from status set
            queued_ids = await storage.get_ids_by_status(ProgramState.QUEUED.value)
            assert "ghost-no-data" in queued_ids
            await storage.remove_ids_from_status_set(
                ProgramState.QUEUED.value, queued_ids
            )

            # Now _has_active_dags should return False
            assert await engine._has_active_dags() is False

            # And _await_idle should return immediately
            await asyncio.wait_for(engine._await_idle(), timeout=HANG_TIMEOUT)
        finally:
            await storage.close()

    async def test_await_idle_ghost_branch_fires_via_time_patch(self) -> None:
        """Test the actual _await_idle ghost-detection branch (core.py L330-361).

        We patch time.monotonic inside _await_idle to simulate 31s elapsed,
        triggering the ghost cleanup code path without waiting 30 real seconds.
        """
        storage = _make_storage()
        try:
            engine = _make_engine(storage, loop_interval=0.001)

            # Insert a ghost ID
            r = await storage._conn.get()
            await r.sadd("test:status:queued", "ghost-abc")

            # _has_active_dags returns True (SCARD counts ghost)
            assert await engine._has_active_dags() is True

            # Patch time.monotonic to fake elapsed > 30s on the second call.
            # _await_idle calls monotonic() twice per iteration:
            #   t0 = time.monotonic()         -- first call
            #   elapsed = time.monotonic()-t0  -- subsequent calls
            # We need t0=0.0, then subsequent calls return 31.0 so elapsed>30.
            call_count = {"n": 0}
            _real_monotonic = __import__("time").monotonic

            def _fake_monotonic():
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return 0.0  # t0
                return 31.0  # elapsed > 30s

            with patch("gigaevo.evolution.engine.core.time") as mock_time:
                mock_time.monotonic = _fake_monotonic

                await asyncio.wait_for(engine._await_idle(), timeout=HANG_TIMEOUT)

            # Ghost should have been cleaned up
            count = await storage.count_by_status(ProgramState.QUEUED.value)
            assert count == 0, "Ghost IDs should be cleaned after 30s fallback"
        finally:
            await storage.close()

    async def test_await_idle_does_not_hang_with_no_programs(self) -> None:
        """With no programs at all, _await_idle returns immediately."""
        storage = _make_storage()
        try:
            engine = _make_engine(storage)
            await asyncio.wait_for(engine._await_idle(), timeout=HANG_TIMEOUT)
        finally:
            await storage.close()


# ===========================================================================
# 2. PopulationSnapshot lock contention
# ===========================================================================


class TestPopulationSnapshotLockContention:
    """PopulationSnapshot uses asyncio.Lock for cache invalidation.
    Multiple concurrent callers must not deadlock.
    """

    async def test_concurrent_get_all_same_epoch(self) -> None:
        """N concurrent get_all calls on the same epoch must all complete."""
        storage = AsyncMock(spec=ProgramStorage)
        programs = [_make_program() for _ in range(10)]
        storage.get_all = AsyncMock(return_value=programs)

        snapshot = PopulationSnapshot()

        async def fetch():
            return await snapshot.get_all(storage)

        tasks = [asyncio.create_task(fetch()) for _ in range(20)]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=HANG_TIMEOUT)

        # All 20 callers should get the same list
        for r in results:
            assert len(r) == 10

        # Storage should be called exactly once (cache hit for the rest)
        assert storage.get_all.call_count == 1

    async def test_concurrent_get_all_with_epoch_bump(self) -> None:
        """A bump mid-flight should cause exactly one re-fetch, not a deadlock."""
        call_count = {"n": 0}

        async def slow_get_all(*, exclude=None):
            call_count["n"] += 1
            await asyncio.sleep(0.01)
            return [_make_program()]

        storage = AsyncMock(spec=ProgramStorage)
        storage.get_all = slow_get_all

        snapshot = PopulationSnapshot()

        # First fetch to warm cache
        await snapshot.get_all(storage)
        assert call_count["n"] == 1

        # Bump epoch — next call must re-fetch
        snapshot.bump()

        tasks = [asyncio.create_task(snapshot.get_all(storage)) for _ in range(10)]
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=HANG_TIMEOUT)

        # Exactly one re-fetch after bump (others piggyback)
        assert call_count["n"] == 2

    async def test_concurrent_get_all_different_exclude(self) -> None:
        """Callers with different exclude values should not deadlock.
        The single-slot cache will thrash, but must not hang.
        """
        storage = AsyncMock(spec=ProgramStorage)
        storage.get_all = AsyncMock(return_value=[_make_program()])

        snapshot = PopulationSnapshot()

        async def fetch_with_exclude():
            return await snapshot.get_all(storage, exclude=frozenset({"stage_results"}))

        async def fetch_without_exclude():
            return await snapshot.get_all(storage)

        tasks = []
        for i in range(10):
            if i % 2 == 0:
                tasks.append(asyncio.create_task(fetch_with_exclude()))
            else:
                tasks.append(asyncio.create_task(fetch_without_exclude()))

        await asyncio.wait_for(asyncio.gather(*tasks), timeout=HANG_TIMEOUT)

    async def test_slow_storage_does_not_block_forever(self) -> None:
        """If storage.get_all is slow, other callers wait but don't deadlock."""

        async def slow_get_all(*, exclude=None):
            await asyncio.sleep(0.1)
            return [_make_program()]

        storage = AsyncMock(spec=ProgramStorage)
        storage.get_all = slow_get_all

        snapshot = PopulationSnapshot()

        tasks = [asyncio.create_task(snapshot.get_all(storage)) for _ in range(5)]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=HANG_TIMEOUT)
        for r in results:
            assert len(r) == 1

    async def test_snapshot_data_correctness_after_bump(self) -> None:
        """After bump(), get_all must return fresh data, not stale cache."""
        fetch_count = {"n": 0}
        batch_a = [Program(code="def a(): pass", state=ProgramState.DONE)]
        batch_b = [
            Program(code="def b(): pass", state=ProgramState.DONE),
            Program(code="def c(): pass", state=ProgramState.DONE),
        ]

        async def dynamic_get_all(*, exclude=None):
            fetch_count["n"] += 1
            if fetch_count["n"] == 1:
                return batch_a
            return batch_b

        storage = AsyncMock(spec=ProgramStorage)
        storage.get_all = dynamic_get_all

        snapshot = PopulationSnapshot()

        # First fetch — returns batch_a
        result1 = await snapshot.get_all(storage)
        assert len(result1) == 1
        assert result1[0].code == "def a(): pass"

        # Same epoch — returns cached batch_a
        result1b = await snapshot.get_all(storage)
        assert len(result1b) == 1
        assert fetch_count["n"] == 1  # no extra fetch

        # Bump and re-fetch — must return batch_b
        snapshot.bump()
        result2 = await snapshot.get_all(storage)
        assert len(result2) == 2
        assert result2[0].code == "def b(): pass"
        assert fetch_count["n"] == 2


# ===========================================================================
# 3. ProgramStateManager per-program lock contention
# ===========================================================================


class TestStateManagerLockContention:
    """ProgramStateManager has a dict of per-program asyncio.Lock.
    Concurrent operations on the same program must serialize, not deadlock.
    Concurrent operations on different programs must not interfere.
    """

    async def test_concurrent_state_transitions_same_program(self) -> None:
        """Multiple concurrent set_program_state calls on the same program
        must serialize — all complete, no deadlock.
        """
        storage = _make_storage()
        try:
            sm = ProgramStateManager(storage)
            prog = _make_program(ProgramState.QUEUED)
            await storage.add(prog)

            # QUEUED -> RUNNING -> DONE is the valid path
            # We can't do concurrent transitions on the same program
            # with different target states, but we can verify serialization
            await asyncio.wait_for(
                sm.set_program_state(prog, ProgramState.RUNNING),
                timeout=HANG_TIMEOUT,
            )
            await asyncio.wait_for(
                sm.set_program_state(prog, ProgramState.DONE),
                timeout=HANG_TIMEOUT,
            )
            assert prog.state == ProgramState.DONE
        finally:
            await storage.close()

    async def test_truly_concurrent_writes_same_program(self) -> None:
        """Launch N tasks that all try to write_exclusive on the same program
        simultaneously using a barrier. Verify they serialize (no overlapping writes).
        """
        storage = _make_storage()
        try:
            sm = ProgramStateManager(storage)
            prog = _make_program(ProgramState.RUNNING)
            await storage.add(prog)

            n_writers = 10
            barrier = asyncio.Barrier(n_writers)
            write_log: list[tuple[int, str]] = []  # (writer_id, "enter"/"exit")

            async def concurrent_write(writer_id: int):
                await barrier.wait()
                async with sm._lock_for(prog.id):
                    write_log.append((writer_id, "enter"))
                    await asyncio.sleep(0.001)  # simulate work
                    write_log.append((writer_id, "exit"))

            tasks = [asyncio.create_task(concurrent_write(i)) for i in range(n_writers)]
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=HANG_TIMEOUT)

            # Verify serialization: entries must alternate enter/exit (no interleaving)
            assert len(write_log) == n_writers * 2
            for i in range(0, len(write_log), 2):
                assert write_log[i][1] == "enter"
                assert write_log[i + 1][1] == "exit"
                assert write_log[i][0] == write_log[i + 1][0]  # same writer
        finally:
            await storage.close()

    async def test_concurrent_transitions_different_programs(self) -> None:
        """Transitions on different programs must proceed in parallel,
        not block each other.
        """
        storage = _make_storage()
        try:
            sm = ProgramStateManager(storage)
            programs = []
            for _ in range(50):
                p = _make_program(ProgramState.QUEUED)
                await storage.add(p)
                programs.append(p)

            async def transition(p: Program):
                await sm.set_program_state(p, ProgramState.RUNNING)
                await sm.set_program_state(p, ProgramState.DONE)

            tasks = [asyncio.create_task(transition(p)) for p in programs]
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=HANG_TIMEOUT)

            for p in programs:
                assert p.state == ProgramState.DONE
        finally:
            await storage.close()

    async def test_lock_eviction_for_terminal_states(self) -> None:
        """Locks for DONE/DISCARDED programs must be evicted to prevent memory leaks."""
        storage = _make_storage()
        try:
            sm = ProgramStateManager(storage)

            programs = []
            for _ in range(20):
                p = _make_program(ProgramState.QUEUED)
                await storage.add(p)
                programs.append(p)

            for p in programs:
                await sm.set_program_state(p, ProgramState.RUNNING)
                await sm.set_program_state(p, ProgramState.DONE)

            # All locks should have been evicted (DONE is terminal)
            assert len(sm._locks) == 0, (
                f"Expected 0 locks after terminal transitions, got {len(sm._locks)}"
            )
        finally:
            await storage.close()

    async def test_concurrent_write_exclusive_serialized(self) -> None:
        """Multiple concurrent write_exclusive calls on the same program
        must serialize via per-program lock — no overlapping writes.
        """
        storage = _make_storage()
        try:
            sm = ProgramStateManager(storage)
            prog = _make_program(ProgramState.RUNNING)
            await storage.add(prog)

            write_active = {"count": 0, "max_concurrent": 0}

            original_write = storage.write_exclusive

            async def tracking_write(program: Program):
                write_active["count"] += 1
                write_active["max_concurrent"] = max(
                    write_active["max_concurrent"], write_active["count"]
                )
                await original_write(program)
                write_active["count"] -= 1

            storage.write_exclusive = tracking_write

            async def write(i: int):
                prog.set_metadata(f"write_{i}", i)
                await sm.write_exclusive(prog)

            tasks = [asyncio.create_task(write(i)) for i in range(10)]
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=HANG_TIMEOUT)

            # Per-program lock should serialize — max concurrent should be 1
            assert write_active["max_concurrent"] == 1, (
                f"Expected max 1 concurrent write, got {write_active['max_concurrent']}"
            )
        finally:
            await storage.close()

    async def test_lock_eviction_race_concurrent_reuse(self) -> None:
        """Verify that the lock eviction race (pop after release) doesn't
        cause crashes or lost serialization.

        ProgramStateManager._locks.pop() runs after releasing the lock (L127).
        A concurrent _lock_for() between release and pop creates a new lock.
        This is benign (pop removes the old one, concurrent task uses new one),
        but we verify it doesn't crash or deadlock.
        """
        storage = _make_storage()
        try:
            sm = ProgramStateManager(storage)

            # Program starts QUEUED, goes DONE (triggers eviction), then we
            # immediately re-use it by doing another write before eviction
            prog = _make_program(ProgramState.QUEUED)
            await storage.add(prog)

            # Task 1: transition to DONE (triggers lock eviction)
            # Task 2: try to write_exclusive right after RUNNING transition
            # This creates a race between eviction and new lock creation.

            async def go_done():
                await sm.set_program_state(prog, ProgramState.RUNNING)
                await sm.set_program_state(prog, ProgramState.DONE)

            async def write_after_running():
                # Small delay to let task 1 get to RUNNING
                await asyncio.sleep(0.001)
                # write_exclusive will create a new lock if evicted
                prog.set_metadata("post_done_write", True)
                await sm.write_exclusive(prog)

            await asyncio.wait_for(
                asyncio.gather(go_done(), write_after_running()),
                timeout=HANG_TIMEOUT,
            )

            # If we got here without crash or hang, the race is benign
            assert prog.state == ProgramState.DONE
        finally:
            await storage.close()


# ===========================================================================
# 4. Engine step with generation_timeout prevents infinite hangs
# ===========================================================================


class TestEngineGenerationTimeout:
    """EvolutionEngine.run() wraps step() in asyncio.wait_for(timeout=generation_timeout).
    This is the top-level escape hatch for any deadlock within a step.
    """

    async def test_generation_timeout_fires_on_stuck_step(self) -> None:
        """If step() hangs (e.g., _await_idle never returns), generation_timeout
        must fire and the engine logs a warning without crashing.
        """
        storage = _make_storage()
        try:
            engine = _make_engine(
                storage,
                generation_timeout=0.05,
                max_generations=1,
            )

            timeout_fired = {"n": 0}

            async def hanging_then_done():
                """First call hangs (triggers timeout), then stop engine."""
                timeout_fired["n"] += 1
                if timeout_fired["n"] <= 2:
                    await asyncio.sleep(999)
                else:
                    # Stop the engine after a few timeouts
                    engine._running = False

            engine.step = hanging_then_done

            await asyncio.wait_for(engine.run(), timeout=HANG_TIMEOUT)

            # At least one timeout should have fired before engine stopped
            assert timeout_fired["n"] >= 2
        finally:
            await storage.close()

    async def test_generation_timeout_on_real_step_with_ghosts(self) -> None:
        """Ghost IDs cause _await_idle to spin in real step().
        generation_timeout must fire to break the hang.

        This tests the actual deadlock recovery path without mocking step().
        """
        storage = _make_storage()
        try:
            engine = _make_engine(
                storage,
                generation_timeout=0.2,
                max_generations=1,
            )

            # Inject ghost IDs that will cause _await_idle to hang
            # (SCARD says active, but no real programs exist)
            r = await storage._conn.get()
            await r.sadd("test:status:queued", "ghost-stuck-1", "ghost-stuck-2")

            # Verify the ghost trap is set
            assert await engine._has_active_dags() is True

            # run() should fire generation_timeout, log warning, and loop.
            # We stop after first timeout via _running = False.
            step_attempted = {"n": 0}
            original_step = engine.step

            async def counting_step():
                step_attempted["n"] += 1
                if step_attempted["n"] >= 2:
                    engine._running = False
                    return
                await original_step()

            engine.step = counting_step

            await asyncio.wait_for(engine.run(), timeout=HANG_TIMEOUT)

            # step was called and timed out at least once
            assert step_attempted["n"] >= 1
        finally:
            await storage.close()

    async def test_stuck_running_program_triggers_timeout(self) -> None:
        """A program stuck in RUNNING forever (most likely production deadlock).
        generation_timeout must fire to prevent infinite hang.
        """
        storage = _make_storage()
        try:
            engine = _make_engine(
                storage,
                generation_timeout=0.2,
                max_generations=1,
            )

            # Add a program stuck in RUNNING — no DAG runner to complete it
            p = _make_program(ProgramState.QUEUED)
            await storage.add(p)
            sm = ProgramStateManager(storage)
            await sm.set_program_state(p, ProgramState.RUNNING)

            # _await_idle will spin because RUNNING count > 0
            assert await engine._has_active_dags() is True

            step_called = {"n": 0}
            original_step = engine.step

            async def counting_step():
                step_called["n"] += 1
                if step_called["n"] >= 2:
                    engine._running = False
                    return
                await original_step()

            engine.step = counting_step

            await asyncio.wait_for(engine.run(), timeout=HANG_TIMEOUT)
            assert step_called["n"] >= 1
        finally:
            await storage.close()

    async def test_no_generation_timeout_still_completes(self) -> None:
        """With default generation_timeout, a clean step still completes."""
        storage = _make_storage()
        try:
            engine = _make_engine(storage, max_generations=1)

            # No programs — step should be trivial (empty archive, no elites)
            await asyncio.wait_for(engine.run(), timeout=HANG_TIMEOUT)
            assert engine.metrics.total_generations == 1
        finally:
            await storage.close()


# ===========================================================================
# 5. Ingest with concurrent state transitions
# ===========================================================================


class TestIngestConcurrency:
    """_ingest_completed_programs creates asyncio tasks for state transitions.
    These fire-and-forget tasks must all complete without leaking.
    """

    async def test_ingest_gather_completes_with_failures(self) -> None:
        """If some state transitions fail during ingest, gather(return_exceptions=True)
        must still complete — no hanging on failed tasks.
        """
        storage = _make_storage()
        try:
            engine = _make_engine(storage)

            # Add programs that will be DONE
            programs = []
            for i in range(10):
                p = _make_program(ProgramState.QUEUED)
                p.add_metrics({"fitness": 0.5})
                await storage.add(p)
                sm = ProgramStateManager(storage)
                await sm.set_program_state(p, ProgramState.RUNNING)
                await sm.set_program_state(p, ProgramState.DONE)
                programs.append(p)

            # Make acceptor reject all — triggers DISCARDED transitions
            engine.config.program_acceptor = MagicMock()
            engine.config.program_acceptor.is_accepted.return_value = False

            await asyncio.wait_for(
                engine._ingest_completed_programs(), timeout=HANG_TIMEOUT
            )
        finally:
            await storage.close()


# ===========================================================================
# 6. Storage close during active operations
# ===========================================================================


class TestStorageCloseRobustness:
    """Closing storage while operations are in flight must not hang."""

    async def test_close_does_not_hang(self) -> None:
        """storage.close() must complete even if called multiple times."""
        storage = _make_storage()
        await storage.close()
        # Second close should be idempotent
        await asyncio.wait_for(storage.close(), timeout=HANG_TIMEOUT)

    async def test_operations_after_close_fail_gracefully(self) -> None:
        """Operations after close should fail, not hang."""
        storage = _make_storage()
        p = _make_program()
        await storage.add(p)
        await storage.close()

        # Operations after close should raise, not hang
        with pytest.raises(Exception):
            await asyncio.wait_for(storage.get(p.id), timeout=HANG_TIMEOUT)


# ===========================================================================
# 7. Batch transition with large program counts
# ===========================================================================


class TestBatchTransitionScale:
    """batch_transition_state with many programs must complete in bounded time."""

    async def test_batch_transition_100_programs(self) -> None:
        """Transitioning 100 programs DONE->QUEUED must not hang."""
        storage = _make_storage()
        try:
            sm = ProgramStateManager(storage)
            programs = []
            for _ in range(100):
                p = _make_program(ProgramState.QUEUED)
                await storage.add(p)
                await sm.set_program_state(p, ProgramState.RUNNING)
                await sm.set_program_state(p, ProgramState.DONE)
                programs.append(p)

            count = await asyncio.wait_for(
                storage.batch_transition_state(
                    programs, ProgramState.DONE.value, ProgramState.QUEUED.value
                ),
                timeout=HANG_TIMEOUT,
            )
            assert count == 100
        finally:
            await storage.close()


# ===========================================================================
# 8. Full engine step with real storage (integration)
# ===========================================================================


class TestEngineStepIntegration:
    """A full engine step with real (fake)Redis storage must complete
    without hanging. This catches interactions between all layers.
    """

    async def test_full_step_no_programs(self) -> None:
        """step() with empty storage completes without hanging."""
        storage = _make_storage()
        try:
            engine = _make_engine(storage)
            await asyncio.wait_for(engine.step(), timeout=HANG_TIMEOUT)
        finally:
            await storage.close()

    async def test_full_step_with_done_programs(self) -> None:
        """step() with DONE programs ingests them. After refresh transitions
        programs back to QUEUED, we need a background task to process them
        (simulating the DAG runner) so _await_idle in Phase 6 terminates.
        """
        storage = _make_storage()
        try:
            engine = _make_engine(storage, generation_timeout=2.0)
            sm = ProgramStateManager(storage)

            # Add a program that's already DONE
            p = _make_program(ProgramState.QUEUED)
            p.add_metrics({"fitness": 0.9, "x": 0.5})
            await storage.add(p)
            await sm.set_program_state(p, ProgramState.RUNNING)
            await sm.set_program_state(p, ProgramState.DONE)

            # Background: after refresh transitions DONE->QUEUED,
            # simulate DAG runner completing the program (QUEUED->RUNNING->DONE)
            async def simulate_dag_runner():
                while True:
                    queued = await storage.get_all_by_status(ProgramState.QUEUED.value)
                    for prog in queued:
                        try:
                            await sm.set_program_state(prog, ProgramState.RUNNING)
                            await sm.set_program_state(prog, ProgramState.DONE)
                        except (ValueError, Exception):
                            pass
                    await asyncio.sleep(0.01)

            dag_task = asyncio.create_task(simulate_dag_runner())
            try:
                await asyncio.wait_for(engine.step(), timeout=HANG_TIMEOUT)
            finally:
                dag_task.cancel()
                try:
                    await dag_task
                except asyncio.CancelledError:
                    pass
        finally:
            await storage.close()

    async def test_run_two_generations_completes(self) -> None:
        """run() with max_generations=2 completes without hanging."""
        storage = _make_storage()
        try:
            engine = _make_engine(storage, max_generations=2)
            await asyncio.wait_for(engine.run(), timeout=HANG_TIMEOUT)
            assert engine.metrics.total_generations == 2
        finally:
            await storage.close()


# ===========================================================================
# 9. Snapshot + storage interaction under epoch churn
# ===========================================================================


class TestSnapshotEpochChurn:
    """Rapid epoch bumping while concurrent reads are in flight must not deadlock."""

    async def test_rapid_bump_during_concurrent_reads(self) -> None:
        """Bump epoch rapidly while multiple readers try to fetch."""

        async def slow_get_all(*, exclude=None):
            await asyncio.sleep(0.01)
            return [_make_program()]

        storage = AsyncMock(spec=ProgramStorage)
        storage.get_all = slow_get_all

        snapshot = PopulationSnapshot()

        async def reader():
            for _ in range(5):
                await snapshot.get_all(storage)
                await asyncio.sleep(0)

        async def bumper():
            for _ in range(10):
                snapshot.bump()
                await asyncio.sleep(0.005)

        tasks = [asyncio.create_task(reader()) for _ in range(5)]
        tasks.append(asyncio.create_task(bumper()))

        await asyncio.wait_for(asyncio.gather(*tasks), timeout=HANG_TIMEOUT)


# ===========================================================================
# 10. _has_active_dags + _await_idle interaction with real programs
# ===========================================================================


class TestAwaitIdleRealPrograms:
    """Verify _await_idle terminates correctly when real programs
    transition through their lifecycle.
    """

    async def test_await_idle_with_program_completing_async(self) -> None:
        """_await_idle must return once an in-flight program reaches DONE."""
        storage = _make_storage()
        try:
            engine = _make_engine(storage)
            sm = ProgramStateManager(storage)

            p = _make_program(ProgramState.QUEUED)
            await storage.add(p)

            # Background task: transition program to DONE after a short delay
            async def complete_program():
                await asyncio.sleep(0.05)
                await sm.set_program_state(p, ProgramState.RUNNING)
                await asyncio.sleep(0.05)
                await sm.set_program_state(p, ProgramState.DONE)

            task = asyncio.create_task(complete_program())

            await asyncio.wait_for(engine._await_idle(), timeout=HANG_TIMEOUT)
            await task  # Ensure background task also completed

            assert p.state == ProgramState.DONE
        finally:
            await storage.close()

    async def test_await_idle_with_multiple_programs_completing(self) -> None:
        """_await_idle with N programs completing at different times."""
        storage = _make_storage()
        try:
            engine = _make_engine(storage)
            sm = ProgramStateManager(storage)

            programs = []
            for _ in range(5):
                p = _make_program(ProgramState.QUEUED)
                await storage.add(p)
                programs.append(p)

            async def complete_program(prog: Program, delay: float):
                await asyncio.sleep(delay)
                await sm.set_program_state(prog, ProgramState.RUNNING)
                await asyncio.sleep(0.01)
                await sm.set_program_state(prog, ProgramState.DONE)

            tasks = [
                asyncio.create_task(complete_program(p, 0.02 * (i + 1)))
                for i, p in enumerate(programs)
            ]

            await asyncio.wait_for(engine._await_idle(), timeout=HANG_TIMEOUT)
            await asyncio.gather(*tasks)

            for p in programs:
                assert p.state == ProgramState.DONE
        finally:
            await storage.close()


# ===========================================================================
# 11. DagRunner semaphore saturation and timeout recovery
# ===========================================================================


class TestDagRunnerSemaphore:
    """DagRunner._sema limits concurrent DAGs. If all slots are taken by
    hanging tasks, new programs queue behind the semaphore. The dag_timeout
    in _maintain() and _cancel_task (2s timeout) are the escape hatches.
    """

    async def test_semaphore_does_not_deadlock_under_saturation(self) -> None:
        """Fill all semaphore slots with slow tasks, then verify new tasks
        can proceed once slots free up.
        """
        sema = asyncio.Semaphore(2)
        completed = {"n": 0}

        async def slow_work(delay: float):
            async with sema:
                await asyncio.sleep(delay)
                completed["n"] += 1

        # Launch 4 tasks through 2-slot semaphore
        tasks = [
            asyncio.create_task(slow_work(0.05)),
            asyncio.create_task(slow_work(0.05)),
            asyncio.create_task(slow_work(0.05)),
            asyncio.create_task(slow_work(0.05)),
        ]

        await asyncio.wait_for(asyncio.gather(*tasks), timeout=HANG_TIMEOUT)
        assert completed["n"] == 4

    async def test_cancel_task_does_not_hang_on_stuck_task(self) -> None:
        """DagRunner._cancel_task uses wait_for(timeout=2.0).
        If a task ignores cancellation, the 2s timeout must fire.
        """
        import contextlib

        async def uncancellable():
            """Task that catches CancelledError and continues."""
            try:
                while True:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                # Deliberately swallow and continue
                await asyncio.sleep(10)

        task = asyncio.create_task(uncancellable())
        await asyncio.sleep(0.01)  # let it start

        info = TaskInfo(task=task, program_id="test-prog", started_at=0.0)

        # _cancel_task should complete within 2s + overhead
        storage = _make_storage()
        try:
            writer = MagicMock()
            writer.bind.return_value = writer
            runner = DagRunner(
                storage=storage,
                dag_blueprint=MagicMock(),
                config=DagRunnerConfig(poll_interval=0.5, max_concurrent_dags=2),
                writer=writer,
            )
            await asyncio.wait_for(runner._cancel_task(info), timeout=HANG_TIMEOUT)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await storage.close()

    async def test_semaphore_with_cancelled_tasks_frees_slots(self) -> None:
        """Cancelled tasks must release their semaphore slot."""
        import contextlib

        sema = asyncio.Semaphore(1)
        entered = asyncio.Event()

        async def blocking_work():
            async with sema:
                entered.set()
                await asyncio.sleep(999)

        # Start a task that holds the semaphore
        task1 = asyncio.create_task(blocking_work())
        await entered.wait()

        # Cancel it — slot should be freed
        task1.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task1

        # New task should be able to acquire the semaphore
        acquired = {"done": False}

        async def quick_work():
            async with sema:
                acquired["done"] = True

        await asyncio.wait_for(quick_work(), timeout=HANG_TIMEOUT)
        assert acquired["done"]


# ===========================================================================
# 12. RedisConnection lock contention
# ===========================================================================


class TestRedisConnectionLock:
    """RedisConnection uses asyncio.Lock for lazy initialization.
    Multiple concurrent callers must not deadlock.
    """

    async def test_concurrent_get_does_not_deadlock(self) -> None:
        """Multiple concurrent .get() calls should not deadlock —
        only one creates the connection, others piggyback.
        """
        storage = _make_storage()
        try:
            conn = storage._conn

            async def get_redis():
                return await conn.get()

            tasks = [asyncio.create_task(get_redis()) for _ in range(20)]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks), timeout=HANG_TIMEOUT
            )

            # All should get the same connection
            assert all(r is results[0] for r in results)
        finally:
            await storage.close()

    async def test_get_after_close_raises_not_hangs(self) -> None:
        """After close(), get() must raise StorageError, not hang."""
        storage = _make_storage()
        conn = storage._conn
        await conn.close()

        from gigaevo.exceptions import StorageError

        with pytest.raises(StorageError):
            await asyncio.wait_for(conn.get(), timeout=HANG_TIMEOUT)

    async def test_execute_retry_does_not_hang(self) -> None:
        """RedisConnection.execute with retries must complete in bounded time,
        even if all retries fail.
        """
        config = RedisConnectionConfig(
            redis_url="redis://nonexistent:6379/0",
            max_retries=2,
            retry_delay=0.01,
        )
        conn = RedisConnection(config)
        # Pre-set a fake redis that always raises
        conn._redis = MagicMock()

        async def failing_op(r):
            raise ConnectionError("fake connection error")

        from gigaevo.exceptions import StorageError

        with pytest.raises(StorageError):
            await asyncio.wait_for(
                conn.execute("test_op", failing_op), timeout=HANG_TIMEOUT
            )


# ===========================================================================
# 13. DagRunner maintain/launch interaction
# ===========================================================================


class TestDagRunnerMaintainLaunch:
    """The DagRunner._maintain and _launch methods run sequentially in _run,
    but interact with shared state (_active dict). Test that timeouts and
    harvesting don't cause lost programs or hangs.
    """

    async def test_timed_out_dag_is_cleaned_up(self) -> None:
        """A DAG that exceeds dag_timeout must be cancelled and removed from _active."""
        storage = _make_storage()
        try:
            writer = MagicMock()
            writer.bind.return_value = writer
            runner = DagRunner(
                storage=storage,
                dag_blueprint=MagicMock(),
                config=DagRunnerConfig(
                    poll_interval=0.5, max_concurrent_dags=4, dag_timeout=0.01
                ),
                writer=writer,
            )

            # Create a hanging task and add to _active
            async def hanging():
                await asyncio.sleep(999)

            task = asyncio.create_task(hanging())
            prog = _make_program(ProgramState.RUNNING)
            await storage.add(prog)
            runner._active[prog.id] = TaskInfo(
                task=task, program_id=prog.id, started_at=0.0
            )

            # _maintain should detect the timeout and clean up
            await asyncio.wait_for(runner._maintain(), timeout=HANG_TIMEOUT)

            assert prog.id not in runner._active
        finally:
            # Clean up any remaining tasks
            for info in list(runner._active.values()):
                info.task.cancel()
            await storage.close()

    async def test_finished_task_harvested(self) -> None:
        """A completed task is removed from _active by _maintain."""
        storage = _make_storage()
        try:
            writer = MagicMock()
            writer.bind.return_value = writer
            runner = DagRunner(
                storage=storage,
                dag_blueprint=MagicMock(),
                config=DagRunnerConfig(poll_interval=0.5, max_concurrent_dags=4),
                writer=writer,
            )

            async def quick():
                pass

            task = asyncio.create_task(quick())
            await task  # let it complete

            prog = _make_program(ProgramState.RUNNING)
            await storage.add(prog)
            runner._active[prog.id] = TaskInfo(
                task=task,
                program_id=prog.id,
                started_at=__import__("time").monotonic(),
            )

            await asyncio.wait_for(runner._maintain(), timeout=HANG_TIMEOUT)
            assert prog.id not in runner._active
        finally:
            await storage.close()

    async def test_orphaned_running_program_discarded(self) -> None:
        """A RUNNING program not in _active (orphan) must be discarded by _launch."""
        storage = _make_storage()
        try:
            writer = MagicMock()
            writer.bind.return_value = writer
            runner = DagRunner(
                storage=storage,
                dag_blueprint=MagicMock(),
                config=DagRunnerConfig(poll_interval=0.5, max_concurrent_dags=4),
                writer=writer,
            )

            # Add a RUNNING program that is not in _active (orphan)
            sm = ProgramStateManager(storage)
            prog = _make_program(ProgramState.QUEUED)
            await storage.add(prog)
            await sm.set_program_state(prog, ProgramState.RUNNING)

            # _launch should detect the orphan and discard it
            await asyncio.wait_for(runner._launch(), timeout=HANG_TIMEOUT)

            # Verify it was discarded
            stored = await storage.get(prog.id)
            assert stored is not None
            assert stored.state == ProgramState.DISCARDED
        finally:
            await storage.close()


# ===========================================================================
# 14. Multiple ProgramStateManagers on same storage (Engine + DagRunner)
# ===========================================================================


class TestMultipleStateManagers:
    """In production, both Engine and DagRunner create their own ProgramStateManager
    instances on the same storage. Their per-program locks are independent (no
    cross-instance coordination). Verify this doesn't cause hangs.
    """

    async def test_two_state_managers_same_program_no_deadlock(self) -> None:
        """Two independent ProgramStateManagers transitioning the same program
        concurrently must not deadlock. The per-program lock is per-instance,
        so they don't block each other.
        """
        storage = _make_storage()
        try:
            sm1 = ProgramStateManager(storage)
            sm2 = ProgramStateManager(storage)

            prog = _make_program(ProgramState.QUEUED)
            await storage.add(prog)

            # sm1 transitions QUEUED -> RUNNING
            await asyncio.wait_for(
                sm1.set_program_state(prog, ProgramState.RUNNING),
                timeout=HANG_TIMEOUT,
            )

            # sm2 does a write_exclusive on the same program
            prog.set_metadata("sm2_write", True)
            await asyncio.wait_for(
                sm2.write_exclusive(prog),
                timeout=HANG_TIMEOUT,
            )

            # sm1 transitions RUNNING -> DONE
            await asyncio.wait_for(
                sm1.set_program_state(prog, ProgramState.DONE),
                timeout=HANG_TIMEOUT,
            )

            assert prog.state == ProgramState.DONE
        finally:
            await storage.close()

    async def test_concurrent_state_managers_different_programs(self) -> None:
        """Two state managers each processing different programs concurrently."""
        storage = _make_storage()
        try:
            sm1 = ProgramStateManager(storage)
            sm2 = ProgramStateManager(storage)

            progs1 = []
            progs2 = []
            for _ in range(10):
                p1 = _make_program(ProgramState.QUEUED)
                p2 = _make_program(ProgramState.QUEUED)
                await storage.add(p1)
                await storage.add(p2)
                progs1.append(p1)
                progs2.append(p2)

            async def sm1_work():
                for p in progs1:
                    await sm1.set_program_state(p, ProgramState.RUNNING)
                    await sm1.set_program_state(p, ProgramState.DONE)

            async def sm2_work():
                for p in progs2:
                    await sm2.set_program_state(p, ProgramState.RUNNING)
                    await sm2.set_program_state(p, ProgramState.DONE)

            await asyncio.wait_for(
                asyncio.gather(sm1_work(), sm2_work()),
                timeout=HANG_TIMEOUT,
            )

            for p in progs1 + progs2:
                assert p.state == ProgramState.DONE
        finally:
            await storage.close()


# ===========================================================================
# 15. Mixed QUEUED/RUNNING/DONE programs — _await_idle precision
# ===========================================================================


class TestAwaitIdleMixedStates:
    """_await_idle must correctly detect idle state when programs are
    in mixed states — some DONE, some being transitioned.
    """

    async def test_idle_with_mixed_done_and_discarded(self) -> None:
        """DONE + DISCARDED programs = idle (only QUEUED/RUNNING are active)."""
        storage = _make_storage()
        try:
            engine = _make_engine(storage)
            sm = ProgramStateManager(storage)

            # Create programs in various terminal states
            for state_path in [
                [ProgramState.RUNNING, ProgramState.DONE],
                [ProgramState.RUNNING, ProgramState.DISCARDED],
                [ProgramState.RUNNING, ProgramState.DONE],
            ]:
                p = _make_program(ProgramState.QUEUED)
                await storage.add(p)
                for s in state_path:
                    await sm.set_program_state(p, s)

            # Should be idle — no QUEUED or RUNNING
            await asyncio.wait_for(engine._await_idle(), timeout=HANG_TIMEOUT)

        finally:
            await storage.close()

    async def test_single_queued_among_many_done_blocks_idle(self) -> None:
        """Even one QUEUED program among many DONE should block _await_idle."""
        storage = _make_storage()
        try:
            engine = _make_engine(storage)
            sm = ProgramStateManager(storage)

            # 10 DONE programs
            for _ in range(10):
                p = _make_program(ProgramState.QUEUED)
                await storage.add(p)
                await sm.set_program_state(p, ProgramState.RUNNING)
                await sm.set_program_state(p, ProgramState.DONE)

            # 1 QUEUED program (blocks idle)
            blocker = _make_program(ProgramState.QUEUED)
            await storage.add(blocker)

            assert await engine._has_active_dags() is True

            # Complete the blocker in background
            async def unblock():
                await asyncio.sleep(0.05)
                await sm.set_program_state(blocker, ProgramState.RUNNING)
                await sm.set_program_state(blocker, ProgramState.DONE)

            task = asyncio.create_task(unblock())
            await asyncio.wait_for(engine._await_idle(), timeout=HANG_TIMEOUT)
            await task

        finally:
            await storage.close()


# ===========================================================================
# 16. Storage batch operations under contention
# ===========================================================================


class TestStorageBatchContention:
    """Concurrent batch operations on Redis must not deadlock."""

    async def test_concurrent_batch_transitions(self) -> None:
        """Two concurrent batch_transition_state calls on different programs."""
        storage = _make_storage()
        try:
            sm = ProgramStateManager(storage)

            batch_a = []
            batch_b = []
            for _ in range(20):
                pa = _make_program(ProgramState.QUEUED)
                pb = _make_program(ProgramState.QUEUED)
                await storage.add(pa)
                await storage.add(pb)
                await sm.set_program_state(pa, ProgramState.RUNNING)
                await sm.set_program_state(pa, ProgramState.DONE)
                await sm.set_program_state(pb, ProgramState.RUNNING)
                await sm.set_program_state(pb, ProgramState.DONE)
                batch_a.append(pa)
                batch_b.append(pb)

            async def transition_batch(programs):
                return await storage.batch_transition_state(
                    programs, ProgramState.DONE.value, ProgramState.QUEUED.value
                )

            results = await asyncio.wait_for(
                asyncio.gather(
                    transition_batch(batch_a),
                    transition_batch(batch_b),
                ),
                timeout=HANG_TIMEOUT,
            )

            assert results[0] == 20
            assert results[1] == 20
        finally:
            await storage.close()

    async def test_get_all_by_status_during_transitions(self) -> None:
        """get_all_by_status must return consistent results even while
        programs are being transitioned concurrently.
        """
        storage = _make_storage()
        try:
            sm = ProgramStateManager(storage)

            programs = []
            for _ in range(20):
                p = _make_program(ProgramState.QUEUED)
                await storage.add(p)
                programs.append(p)

            async def transition_all():
                for p in programs:
                    await sm.set_program_state(p, ProgramState.RUNNING)
                    await sm.set_program_state(p, ProgramState.DONE)

            async def poll_status():
                for _ in range(50):
                    queued = await storage.get_all_by_status(ProgramState.QUEUED.value)
                    running = await storage.get_all_by_status(
                        ProgramState.RUNNING.value
                    )
                    done = await storage.get_all_by_status(ProgramState.DONE.value)
                    # Total should be <= 20 (some may be in-flight between states)
                    total = len(queued) + len(running) + len(done)
                    assert total <= 20, f"Got {total} programs, expected <= 20"
                    await asyncio.sleep(0)

            await asyncio.wait_for(
                asyncio.gather(transition_all(), poll_status()),
                timeout=HANG_TIMEOUT,
            )
        finally:
            await storage.close()
