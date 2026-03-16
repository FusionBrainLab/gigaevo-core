"""Integration tests: EvolutionEngine edge cases and program state transitions.

Area 2 — EvolutionEngine:
  1. Engine with zero seed programs — first generation produces mutations from
     an empty archive (no crash; elites=[] → created=0 → generation still advances).
  2. After max_generations, engine.task completes and
     engine.metrics.total_generations == max_generations.
  3. A generation where all mutations are rejected by the archive (duplicate bins)
     — engine still advances generation counter.

Area 3 — Program state transitions via ProgramStateManager:
  1. Valid transition: QUEUED → RUNNING → DONE (happy path, storage reflects each).
  2. Invalid transition: DONE → RUNNING raises ValueError.
  3. get_all_by_status returns only programs in the requested state.

Setup follows test_evolution_metrics_pipeline.py conventions:
  - FloatHalvingOperator for deterministic mutations.
  - FakeDagRunner loop for synchronous evaluation.
  - fakeredis for storage (no real Redis).
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import re
from unittest.mock import MagicMock

import fakeredis.aioredis
import pytest

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

# ---------------------------------------------------------------------------
# Constants (mirrored from test_resume_e2e / test_evolution_metrics_pipeline)
# ---------------------------------------------------------------------------

SEED_VALUE = 1024.0
SEED_CODE = f"def entrypoint():\n    return {SEED_VALUE}"

_VALUE_RE = re.compile(r"return\s+([\d.]+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_value(code: str) -> float:
    m = _VALUE_RE.search(code)
    if m is None:
        raise ValueError(f"Cannot extract return value from code:\n{code}")
    return float(m.group(1))


def _halved_code(value: float) -> str:
    return f"def entrypoint():\n    return {value / 2.0}"


def _compute_metrics(code: str) -> dict[str, float]:
    """fitness = -value (lower value → higher fitness); x = halving depth."""
    value = _extract_value(code)
    depth = math.log2(SEED_VALUE / value) if value > 0 else 0.0
    return {"fitness": -value, "x": depth}


# ---------------------------------------------------------------------------
# Mutation operator: halves the return value (deterministic)
# ---------------------------------------------------------------------------


class FloatHalvingOperator(MutationOperator):
    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> MutationSpec | None:
        parent = selected_parents[0]
        value = _extract_value(parent.code)
        return MutationSpec(
            code=_halved_code(value),
            parents=selected_parents,
            name="halving",
        )


# ---------------------------------------------------------------------------
# Fake DAG runner (synchronous evaluation in background loop)
# ---------------------------------------------------------------------------


class FakeDagRunner:
    """Evaluates QUEUED programs immediately (sets metrics + DONE)."""

    def __init__(
        self, storage: RedisProgramStorage, state_manager: ProgramStateManager
    ):
        self._storage = storage
        self._sm = state_manager
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="fake-dag-runner")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        while True:
            queued = await self._storage.get_all_by_status(ProgramState.QUEUED.value)
            for prog in queued:
                await self._evaluate(prog)
            await asyncio.sleep(0.005)

    async def _evaluate(self, prog: Program) -> None:
        await self._sm.set_program_state(prog, ProgramState.RUNNING)
        metrics = _compute_metrics(prog.code)
        prog.add_metrics(metrics)
        await self._sm.set_program_state(prog, ProgramState.DONE)


# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------


def _make_fakeredis_storage(server: fakeredis.FakeServer) -> RedisProgramStorage:
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0", key_prefix="test"
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage


def _make_island_config(fitness_key: str = "fitness") -> IslandConfig:
    behavior_space = BehaviorSpace(
        bins={"x": LinearBinning(min_val=0.0, max_val=5.0, num_bins=5, type="linear")}
    )
    return IslandConfig(
        island_id="test",
        behavior_space=behavior_space,
        max_size=None,
        archive_selector=SumArchiveSelector(fitness_keys=[fitness_key]),
        archive_remover=None,
        elite_selector=ScalarTournamentEliteSelector(
            fitness_key=fitness_key,
            fitness_key_higher_is_better=True,
            tournament_size=99,
        ),
        migrant_selector=RandomMigrantSelector(),
    )


def _make_null_writer() -> MagicMock:
    writer = MagicMock()
    writer.bind.return_value = writer
    return writer


def _make_metrics_tracker() -> MagicMock:
    tracker = MagicMock()
    tracker.start = MagicMock()

    async def _stop():
        pass

    tracker.stop = _stop
    return tracker


def _build_engine(
    storage: RedisProgramStorage,
    max_generations: int,
    *,
    fitness_key: str = "fitness",
) -> EvolutionEngine:
    strategy = MapElitesMultiIsland(
        island_configs=[_make_island_config(fitness_key)],
        program_storage=storage,
    )
    return EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=FloatHalvingOperator(),
        config=EngineConfig(
            loop_interval=0.005,
            max_elites_per_generation=1,
            max_mutations_per_generation=1,
            generation_timeout=30.0,
            max_generations=max_generations,
        ),
        writer=_make_null_writer(),
        metrics_tracker=_make_metrics_tracker(),
    )


async def _run_engine(
    storage: RedisProgramStorage,
    max_generations: int,
    *,
    fitness_key: str = "fitness",
) -> EvolutionEngine:
    """Start engine + DAG runner; await task completion."""
    engine = _build_engine(storage, max_generations, fitness_key=fitness_key)
    sm = ProgramStateManager(storage)
    runner = FakeDagRunner(storage, sm)

    runner.start()
    engine.start()
    try:
        await asyncio.wait_for(engine.task, timeout=30.0)
    except TimeoutError:
        pytest.fail(f"Engine did not finish {max_generations} gens within 30s")
    finally:
        await runner.stop()
        await storage.close()

    return engine


# ---------------------------------------------------------------------------
# Area 2, Test 1: Engine with ZERO seed programs
# ---------------------------------------------------------------------------


class TestEmptyArchiveEngine:
    async def test_engine_with_no_seeds_completes_without_crash(self) -> None:
        """Starting the engine with an empty archive must not crash.

        With no elites to select, _create_mutants() is skipped and the
        generation counter still increments once.  This verifies the
        engine's graceful empty-archive handling.
        """
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)
        # Deliberately add NO seed programs.

        engine = await _run_engine(storage, max_generations=1)

        # Engine completed cleanly
        assert engine.metrics.total_generations == 1

    async def test_empty_archive_engine_creates_zero_mutants(self) -> None:
        """With an empty archive, no mutations are created in the first generation."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)

        engine = await _run_engine(storage, max_generations=1)

        # mutations_created == 0 because there were no elites to mutate
        assert engine.metrics.mutations_created == 0

    async def test_engine_with_no_seeds_zero_programs_accepted(self) -> None:
        """Empty archive run: zero programs accepted into strategy."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)

        engine = await _run_engine(storage, max_generations=1)

        assert engine.metrics.programs_processed == 0


# ---------------------------------------------------------------------------
# Area 2, Test 2: Engine stops at max_generations
# ---------------------------------------------------------------------------


class TestMaxGenerationsCap:
    async def test_engine_stops_exactly_at_max_generations(self) -> None:
        """After max_generations steps, engine.task completes and counter matches."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)

        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        engine = await _run_engine(storage, max_generations=3)

        assert engine.metrics.total_generations == 3

    async def test_engine_task_is_done_after_max_generations(self) -> None:
        """engine.task must be done (not cancelled, not running) after cap is reached."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)

        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        engine = _build_engine(storage, max_generations=2)
        sm = ProgramStateManager(storage)
        runner = FakeDagRunner(storage, sm)

        runner.start()
        engine.start()
        try:
            await asyncio.wait_for(engine.task, timeout=30.0)
        except TimeoutError:
            pytest.fail("Engine did not finish within 30s")
        finally:
            await runner.stop()
            await storage.close()

        # Task must be done (not running, not pending)
        assert engine.task is None or engine.task.done(), (
            "engine.task should be done after reaching max_generations"
        )
        assert engine.metrics.total_generations == 2

    async def test_engine_does_not_overshoot_generation_cap(self) -> None:
        """total_generations must never exceed max_generations."""
        server = fakeredis.FakeServer()
        storage = _make_fakeredis_storage(server)

        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        engine = await _run_engine(storage, max_generations=4)

        assert engine.metrics.total_generations <= 4, (
            f"Engine overshot cap: total_generations={engine.metrics.total_generations}"
        )


# ---------------------------------------------------------------------------
# Area 2, Test 3: All mutations rejected (duplicate bins)
# ---------------------------------------------------------------------------


class NullMutationOperator(MutationOperator):
    """Always returns the same code as the parent — guaranteed duplicate bin."""

    async def mutate_single(
        self, selected_parents: list[Program]
    ) -> MutationSpec | None:
        parent = selected_parents[0]
        # Return identical code: same bin, same fitness → archive rejects it
        return MutationSpec(
            code=parent.code,
            parents=selected_parents,
            name="no-op",
        )


class TestAllMutationsRejected:
    async def test_generation_counter_advances_when_all_mutations_rejected(
        self,
    ) -> None:
        """Even if every mutation is rejected (same bin), the generation counter
        must still increment.  This validates the engine does not stall when the
        archive refuses all new programs.
        """
        server = fakeredis.FakeServer()
        config = RedisProgramStorageConfig(
            redis_url="redis://fake:6379/0", key_prefix="test"
        )
        storage = RedisProgramStorage(config)
        fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        storage._conn._redis = fake_redis
        storage._conn._closing = False

        island_config = _make_island_config()
        strategy = MapElitesMultiIsland(
            island_configs=[island_config],
            program_storage=storage,
        )
        engine = EvolutionEngine(
            storage=storage,
            strategy=strategy,
            mutation_operator=NullMutationOperator(),
            config=EngineConfig(
                loop_interval=0.005,
                max_elites_per_generation=1,
                max_mutations_per_generation=1,
                generation_timeout=30.0,
                max_generations=3,
            ),
            writer=_make_null_writer(),
            metrics_tracker=_make_metrics_tracker(),
        )

        sm = ProgramStateManager(storage)
        runner = FakeDagRunner(storage, sm)

        # Seed program: must be in archive before mutations start
        seed = Program(code=SEED_CODE, state=ProgramState.QUEUED)
        await storage.add(seed)

        runner.start()
        engine.start()
        try:
            await asyncio.wait_for(engine.task, timeout=30.0)
        except TimeoutError:
            pytest.fail("Engine stalled when all mutations were rejected")
        finally:
            await runner.stop()
            await storage.close()

        # Generation counter must have advanced despite all rejections
        assert engine.metrics.total_generations == 3, (
            f"Expected 3 generations, got {engine.metrics.total_generations}"
        )


# ---------------------------------------------------------------------------
# Area 3: Program state transitions via ProgramStateManager
# ---------------------------------------------------------------------------


class TestProgramStateTransitions:
    async def test_queued_to_running_to_done_happy_path(
        self, fakeredis_storage, state_manager
    ) -> None:
        """QUEUED → RUNNING → DONE: storage reflects each transition."""
        prog = Program(code="def f(): pass", state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        # Verify initial state in storage
        stored = await fakeredis_storage.get(prog.id)
        assert stored is not None
        assert stored.state == ProgramState.QUEUED

        # QUEUED → RUNNING
        await state_manager.set_program_state(prog, ProgramState.RUNNING)
        assert prog.state == ProgramState.RUNNING

        queued_ids = await fakeredis_storage.get_ids_by_status(
            ProgramState.QUEUED.value
        )
        running_ids = await fakeredis_storage.get_ids_by_status(
            ProgramState.RUNNING.value
        )
        assert prog.id not in queued_ids, "Program should have left QUEUED set"
        assert prog.id in running_ids, "Program should be in RUNNING set"

        # RUNNING → DONE
        await state_manager.set_program_state(prog, ProgramState.DONE)
        assert prog.state == ProgramState.DONE

        running_ids_after = await fakeredis_storage.get_ids_by_status(
            ProgramState.RUNNING.value
        )
        done_ids = await fakeredis_storage.get_ids_by_status(ProgramState.DONE.value)
        assert prog.id not in running_ids_after, "Program should have left RUNNING set"
        assert prog.id in done_ids, "Program should be in DONE set"

    async def test_invalid_transition_done_to_running_raises(
        self, fakeredis_storage, state_manager
    ) -> None:
        """DONE → RUNNING is not a valid transition and must raise ValueError.

        From program_state.py: VALID_TRANSITIONS[DONE] = {QUEUED, DISCARDED}.
        Attempting DONE → RUNNING must be rejected by validate_transition().
        """
        prog = Program(code="def f(): pass", state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        # Advance to DONE via the valid path
        await state_manager.set_program_state(prog, ProgramState.RUNNING)
        await state_manager.set_program_state(prog, ProgramState.DONE)
        assert prog.state == ProgramState.DONE

        # Attempting DONE → RUNNING must raise
        with pytest.raises(ValueError, match="Invalid state transition"):
            await state_manager.set_program_state(prog, ProgramState.RUNNING)

        # State must remain DONE after the failed transition
        assert prog.state == ProgramState.DONE

    async def test_invalid_transition_discarded_to_any_raises(
        self, fakeredis_storage, state_manager
    ) -> None:
        """DISCARDED is a terminal state — no further transitions are valid."""
        prog = Program(code="def f(): pass", state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        await state_manager.set_program_state(prog, ProgramState.DISCARDED)
        assert prog.state == ProgramState.DISCARDED

        # All outgoing transitions from DISCARDED must be rejected
        for bad_target in (
            ProgramState.QUEUED,
            ProgramState.RUNNING,
            ProgramState.DONE,
        ):
            with pytest.raises(ValueError, match="Invalid state transition"):
                await state_manager.set_program_state(prog, bad_target)

    async def test_get_all_by_status_returns_only_requested_state(
        self, fakeredis_storage, state_manager
    ) -> None:
        """get_all_by_status must return only programs in the given state.

        We add 3 programs: 2 QUEUED and 1 RUNNING.
        get_all_by_status(QUEUED) must return exactly the 2 QUEUED programs.
        get_all_by_status(RUNNING) must return exactly the 1 RUNNING program.
        """
        p1 = Program(code="def f(): return 1", state=ProgramState.QUEUED)
        p2 = Program(code="def f(): return 2", state=ProgramState.QUEUED)
        p3 = Program(code="def f(): return 3", state=ProgramState.QUEUED)

        await fakeredis_storage.add(p1)
        await fakeredis_storage.add(p2)
        await fakeredis_storage.add(p3)

        # Advance p3 to RUNNING via state_manager
        await state_manager.set_program_state(p3, ProgramState.RUNNING)

        queued = await fakeredis_storage.get_all_by_status(ProgramState.QUEUED.value)
        running = await fakeredis_storage.get_all_by_status(ProgramState.RUNNING.value)

        queued_ids = {p.id for p in queued}
        running_ids = {p.id for p in running}

        assert queued_ids == {p1.id, p2.id}, (
            f"Expected QUEUED={{p1,p2}}, got ids={queued_ids}"
        )
        assert running_ids == {p3.id}, f"Expected RUNNING={{p3}}, got ids={running_ids}"

    async def test_get_all_by_status_returns_empty_for_unused_state(
        self, fakeredis_storage
    ) -> None:
        """get_all_by_status returns [] for a state with no programs."""
        done = await fakeredis_storage.get_all_by_status(ProgramState.DONE.value)
        assert done == []

    async def test_no_id_leaks_between_status_sets_after_transition(
        self, fakeredis_storage, state_manager
    ) -> None:
        """After a transition, the program's ID must not appear in the old status set.

        This guards against a potential bug where the status set for the old
        state retains the ID after the transition.
        """
        prog = Program(code="def f(): pass", state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        await state_manager.set_program_state(prog, ProgramState.RUNNING)

        # Must not be in QUEUED set after transition
        queued = await fakeredis_storage.get_all_by_status(ProgramState.QUEUED.value)
        queued_ids = {p.id for p in queued}
        assert prog.id not in queued_ids, (
            "Program ID leaked into QUEUED set after transition to RUNNING"
        )

        await state_manager.set_program_state(prog, ProgramState.DONE)

        # Must not be in RUNNING set after transition to DONE
        running = await fakeredis_storage.get_all_by_status(ProgramState.RUNNING.value)
        running_ids = {p.id for p in running}
        assert prog.id not in running_ids, (
            "Program ID leaked into RUNNING set after transition to DONE"
        )
