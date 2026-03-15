"""Regression tests for EvolutionEngine._ingest_completed_programs exception safety.

Bug: the for-loop in _ingest_completed_programs has no per-item exception handling.
If strategy.add() raises for program[0], the exception propagates out of the loop
and programs[1..N] are never processed — they stay in DONE state permanently.
On the next generation the engine picks them up again, potentially looping forever.

Fix: wrap each loop iteration in try/except; on exception log the error, DISCARD
the offending program, and continue processing the remaining items.
"""

from __future__ import annotations

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


def _make_storage() -> tuple[RedisProgramStorage, object]:
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


def _make_engine(storage: RedisProgramStorage) -> EvolutionEngine:
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


async def test_ingest_continues_after_strategy_add_exception() -> None:
    """_ingest_completed_programs must not abort when strategy.add() raises.

    Sequence:
      1. Three programs in DONE state with minimal metrics (pass default acceptor).
      2. strategy.add() is monkeypatched to raise RuntimeError for the first call,
         then succeed for subsequent calls.
      3. _ingest_completed_programs() is called.
      4. Before the fix: the RuntimeError propagates out of the for-loop;
         programs[1] and [2] remain in DONE state (never processed).
      5. After the fix: the exception is caught per-item; programs[1] and [2]
         are processed and leave DONE state (DISCARDED or archived).
    """
    storage, _ = _make_storage()
    try:
        engine = _make_engine(storage)
        sm = ProgramStateManager(storage)

        programs = []
        for i in range(3):
            p = Program(
                code="def solve(): pass",
                state=ProgramState.QUEUED,
                atomic_counter=i,
            )
            await storage.add(p)
            await sm.set_program_state(p, ProgramState.RUNNING)
            await sm.set_program_state(p, ProgramState.DONE)
            p.add_metrics({"fitness": float(i) * 0.1})  # non-empty metrics
            await storage.update(p)
            programs.append(p)

        # Sanity: all three are in DONE state
        for p in programs:
            stored = await storage.get(p.id)
            assert stored is not None and stored.state == ProgramState.DONE

        # Patch strategy.add() to raise on the first call only
        call_count = 0
        original_add = engine.strategy.add

        async def _sometimes_failing_add(prog: Program) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated transient Redis failure in strategy.add")
            return await original_add(prog)

        engine.strategy.add = _sometimes_failing_add

        # Must not raise; must process all three programs
        await engine._ingest_completed_programs()

        # Programs[1] and [2] must NOT still be DONE — they should have been
        # accepted or discarded by the continuation of the loop.
        for i, p in enumerate(programs[1:], start=1):
            final = await storage.get(p.id)
            assert final is not None
            assert final.state != ProgramState.DONE, (
                f"Program[{i}] is still DONE after _ingest_completed_programs().  "
                "Fix: wrap each loop iteration in try/except so a failure on one "
                "program does not abort processing of the remaining ones."
            )
    finally:
        await storage.close()


async def test_ingest_does_not_raise_on_acceptor_exception() -> None:
    """_ingest_completed_programs must survive if is_accepted() raises.

    Same class of bug as the strategy.add() case — no per-item guard.
    """
    storage, _ = _make_storage()
    try:
        engine = _make_engine(storage)
        sm = ProgramStateManager(storage)

        programs = []
        for i in range(2):
            p = Program(
                code="def solve(): pass",
                state=ProgramState.QUEUED,
                atomic_counter=i,
            )
            await storage.add(p)
            await sm.set_program_state(p, ProgramState.RUNNING)
            await sm.set_program_state(p, ProgramState.DONE)
            p.add_metrics({"fitness": 0.5})
            await storage.update(p)
            programs.append(p)

        # Patch acceptor to raise on first call
        call_count = 0
        original_accept = engine.config.program_acceptor.is_accepted

        def _sometimes_failing_acceptor(prog: Program) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated acceptor failure")
            return original_accept(prog)

        engine.config.program_acceptor.is_accepted = _sometimes_failing_acceptor

        # Must not raise
        await engine._ingest_completed_programs()

        # Program[1] must have been processed (not still DONE)
        final = await storage.get(programs[1].id)
        assert final is not None
        assert final.state != ProgramState.DONE, (
            "Program[1] is still DONE — acceptor exception aborted the loop. "
            "Fix: per-item try/except in _ingest_completed_programs."
        )
    finally:
        await storage.close()
