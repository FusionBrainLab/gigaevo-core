"""Round-3 regression tests: prove the two HIGH bugs are fixed.

Bug A — DagRunner._maintain TOCTOU (FIXED):
    _maintain now checks the program's current state after classifying a task
    as timed-out. If the program is already DONE in storage (the task completed
    successfully between classification and the discard write), _maintain skips
    the DISCARD and records the timeout metric without clobbering the result.

Bug B — Ghost IDs in Redis status sets stalling _has_active_dags (FIXED):
    _has_active_dags() now calls get_all_by_status() (SMEMBERS + MGET, ghosts
    filtered) instead of count_by_status() (raw SCARD, ghosts counted).  A
    ghost ID in the QUEUED set therefore no longer causes _await_idle() to loop
    forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import MagicMock

import fakeredis.aioredis

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.strategies.elite_selectors import ScalarTournamentEliteSelector
from gigaevo.evolution.strategies.island import IslandConfig
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.runner.dag_runner import DagRunner, DagRunnerConfig, TaskInfo

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_storage() -> tuple[RedisProgramStorage, object]:
    """Return a RedisProgramStorage backed by an isolated fakeredis server."""
    server = fakeredis.FakeServer()
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix="test",
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage, fake_redis


def _make_program(state: ProgramState = ProgramState.QUEUED) -> Program:
    return Program(
        code="def solve(): return 1", state=state, atomic_counter=999_999_999
    )


def _make_runner(storage: RedisProgramStorage) -> DagRunner:
    dag_blueprint = MagicMock()
    writer = MagicMock()
    writer.bind = MagicMock(return_value=writer)
    return DagRunner(
        storage=storage,
        dag_blueprint=dag_blueprint,
        config=DagRunnerConfig(dag_timeout=1.0, poll_interval=0.1),
        writer=writer,
    )


def _make_engine(storage: RedisProgramStorage) -> EvolutionEngine:
    """Minimal EvolutionEngine for testing _has_active_dags."""

    class _NullMutator(MutationOperator):
        async def mutate_single(
            self, selected_parents: list[Program]
        ) -> MutationSpec | None:
            return None

    strategy = MapElitesMultiIsland(
        island_configs=[
            IslandConfig(
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
        ],
        program_storage=storage,
    )
    tracker = MagicMock()
    tracker.start = MagicMock()

    async def _stop():
        pass

    tracker.stop = _stop

    writer = MagicMock()
    writer.bind.return_value = writer

    return EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=_NullMutator(),
        config=EngineConfig(loop_interval=0.005, max_generations=1),
        writer=writer,
        metrics_tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Bug A — DagRunner._maintain TOCTOU (now fixed)
#
# Fix: _maintain fetches the program from Redis before writing DISCARDED.
# If prog.state == DONE it logs a warning, records the timeout metric, and
# skips the DISCARD (continues to the next timed-out task).
#
# Regression tests verify:
#   (a) A program that is already DONE in Redis is NOT discarded by _maintain.
#   (b) A program that is still RUNNING when it times out IS discarded (normal path).
# ---------------------------------------------------------------------------


async def test_toctou_maintain_skips_discard_for_done_program() -> None:
    """Regression (Bug A fix): _maintain must not discard an already-DONE program.

    Sequence:
      1. Program is QUEUED → RUNNING → DONE in storage
         (simulates _execute_dag completing successfully).
      2. A long-running asyncio task representing the same program is in
         _active with a started_at far in the past (so _maintain classifies
         it as timed-out).
      3. _maintain() is called — without the fix, step 4 below would discard.
      4. Because the program is DONE in Redis, _maintain skips DISCARD.
      5. Final state must still be DONE.
    """
    storage, _ = _make_storage()
    try:
        state_manager = ProgramStateManager(storage)
        runner = _make_runner(storage)

        # Step 1: register and advance program to DONE
        program = _make_program(state=ProgramState.QUEUED)
        await storage.add(program)
        await state_manager.set_program_state(program, ProgramState.RUNNING)
        await state_manager.set_program_state(program, ProgramState.DONE)

        stored = await storage.get(program.id)
        assert stored is not None and stored.state == ProgramState.DONE

        # Step 2: add a sleeping task to _active with an old start time so
        # _maintain classifies it as timed-out (task.done() is False).
        async def _sleeper():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_sleeper())
        old_start = time.monotonic() - 10_000.0  # way past dag_timeout=1.0
        runner._active[program.id] = TaskInfo(task, program.id, old_start)

        # Step 3+4: call _maintain — the guard should kick in.
        await runner._maintain()

        # Step 5: the program must still be DONE.
        final = await storage.get(program.id)
        assert final is not None
        assert final.state == ProgramState.DONE, (
            f"_maintain discarded a successfully-completed program. "
            f"Final state: {final.state}. "
            f"Fix: _maintain must check prog.state == DONE before discarding."
        )

        # Timeout metric should still be recorded (the timeout did happen).
        assert runner._metrics.dag_timeouts == 1

    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await storage.close()


async def test_toctou_maintain_still_discards_running_program_on_timeout() -> None:
    """Regression (Bug A fix): normal timeout path still works.

    If the program is RUNNING (not yet DONE) when the timeout fires, it must
    be discarded as before.  The DONE guard must not break the normal path.
    """
    storage, _ = _make_storage()
    try:
        state_manager = ProgramStateManager(storage)
        runner = _make_runner(storage)

        program = _make_program(state=ProgramState.QUEUED)
        await storage.add(program)
        await state_manager.set_program_state(program, ProgramState.RUNNING)

        stored = await storage.get(program.id)
        assert stored is not None and stored.state == ProgramState.RUNNING

        # Timed-out sleeping task
        async def _sleeper():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_sleeper())
        old_start = time.monotonic() - 10_000.0
        runner._active[program.id] = TaskInfo(task, program.id, old_start)

        await runner._maintain()

        final = await storage.get(program.id)
        assert final is not None
        assert final.state == ProgramState.DISCARDED, (
            f"A timed-out RUNNING program must be DISCARDED. "
            f"Final state: {final.state}."
        )
        assert runner._metrics.dag_timeouts == 1

    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await storage.close()


# ---------------------------------------------------------------------------
# Bug B — Ghost IDs stall _has_active_dags forever (now fixed)
#
# Fix: _has_active_dags() now calls get_all_by_status() (SMEMBERS + MGET,
# filters Nones) instead of count_by_status() (raw SCARD).  Ghost IDs — set
# members with no backing program hash — therefore produce zero real programs
# and _has_active_dags() correctly returns False.
#
# Regression test: inject a ghost into the QUEUED set → _has_active_dags()
# must return False (engine can proceed; _await_idle() terminates).
# ---------------------------------------------------------------------------


async def test_has_active_dags_returns_false_with_ghost_id() -> None:
    """Regression (Bug B fix): _has_active_dags() must return False for ghost IDs.

    A ghost ID is a program ID that exists in a Redis status set but has no
    corresponding program hash in Redis.  Before the fix, _has_active_dags()
    called count_by_status() which uses raw SCARD and counted the ghost,
    returning True forever and stalling _await_idle().

    After the fix, get_all_by_status() filters ghosts out via MGET → Nones
    dropped → real count == 0 → _has_active_dags() returns False.
    """
    storage, fake_redis = _make_storage()
    try:
        engine = _make_engine(storage)

        # Inject a ghost ID directly into the QUEUED status set.
        ghost_id = "ghost-id-no-backing-data-99999"
        ghost_set_key = "test:status:queued"

        r = await storage._conn.get()
        inserted = await r.sadd(ghost_set_key, ghost_id)
        assert inserted == 1, "Ghost must be inserted into the status set"

        # Confirm the ghost has NO backing program hash.
        program_key = f"test:program:{ghost_id}"
        assert await r.exists(program_key) == 0, "Ghost must have no backing hash"

        # _has_active_dags() must return False: the ghost ID has no real program.
        result = await engine._has_active_dags()
        assert result is False, (
            "_has_active_dags() returned True for a ghost ID. "
            "Fix: use get_all_by_status() (MGET-filtered) instead of "
            "count_by_status() (raw SCARD) to exclude ghost IDs."
        )

    finally:
        await storage.close()


async def test_has_active_dags_returns_true_for_real_queued_program() -> None:
    """Sanity: _has_active_dags() returns True when a real QUEUED program exists.

    Ensures the fix did not over-correct — a real program with backing data
    must still be counted and must make _has_active_dags() return True.
    """
    storage, _ = _make_storage()
    try:
        state_manager = ProgramStateManager(storage)
        engine = _make_engine(storage)

        program = _make_program(state=ProgramState.QUEUED)
        await storage.add(program)

        result = await engine._has_active_dags()
        assert result is True, (
            "_has_active_dags() returned False for a real QUEUED program."
        )

        # After marking DONE, should return False again.
        await state_manager.set_program_state(program, ProgramState.RUNNING)
        await state_manager.set_program_state(program, ProgramState.DONE)

        result_after = await engine._has_active_dags()
        assert result_after is False, (
            "_has_active_dags() still True after all programs reached DONE."
        )

    finally:
        await storage.close()
