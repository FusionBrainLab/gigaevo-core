"""Resume-after-kill invariants for the steady-state engine.

Simulates a process kill mid-run by tearing down the engine instance and
constructing a new one bound to the same fakeredis storage. The new
engine calls :meth:`EvolutionEngine.restore_state` to hydrate counters
from :class:`EngineSnapshot`, then runs to a higher mutant cap and we
verify the run continues without restarting from zero.

Key invariants:

- ``EngineSnapshot.total_mutants`` persists across engine teardown +
  reconstruction.
- ``EvolutionEngine.restore_state()`` lifts ``total_mutants`` and
  ``programs_processed`` from the snapshot into the metrics object so
  the stopper does not "forget" prior progress.
- A resumed engine with a higher cap produces additional mutants and
  ends with ``total_mutants ∈ [new_cap, new_cap + max_in_flight]``
  (bounded overshoot, same as the cold-start case).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
import contextlib
from unittest.mock import MagicMock

import fakeredis
import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import MaxMutantsStopper
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.mutation.parent_selector import RandomParentSelector
from gigaevo.evolution.strategies.base import EvolutionStrategy
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Harness (mirrors helpers used in test_engine_cancellation / test_engine_stress)
# ---------------------------------------------------------------------------


def _make_fakeredis_storage(
    server: fakeredis.FakeServer, *, prefix: str = "resume"
) -> RedisProgramStorage:
    """Build a storage instance bound to the given fakeredis server.

    Reusing a single FakeServer across two storage instances is the
    fakeredis-idiomatic way to simulate process death + restart: the
    in-memory store survives even after the first storage instance is
    closed."""
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0", key_prefix=prefix
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage


def _make_null_writer() -> MagicMock:
    writer = MagicMock()
    writer.bind.return_value = writer
    return writer


class _StubStrategy(EvolutionStrategy):
    def __init__(self, seed: list[Program]) -> None:
        self._archive: list[Program] = list(seed)
        self._ids: set[str] = {p.id for p in seed}

    async def add(self, program: Program) -> bool:
        if program.id in self._ids:
            return False
        self._archive.append(program)
        self._ids.add(program.id)
        return True

    async def select_elites(self, total: int) -> list[Program]:
        return list(self._archive)

    async def get_program_ids(self) -> list[str]:
        return list(self._ids)


class _TimedDag:
    """Drives QUEUED → RUNNING → DONE with a small constant delay."""

    def __init__(self, storage: RedisProgramStorage, *, delay: float = 0.003) -> None:
        self._storage = storage
        self._delay = delay
        self._task: asyncio.Task | None = None
        self._claimed: set[str] = set()
        self.flip_count: dict[str, int] = defaultdict(int)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="resume-fake-dag")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        active: set[asyncio.Task] = set()
        try:
            while True:
                queued_ids = await self._storage.get_ids_by_status(
                    ProgramState.QUEUED.value
                )
                fresh = [pid for pid in queued_ids if pid not in self._claimed]
                for pid in fresh:
                    self._claimed.add(pid)
                    t = asyncio.create_task(self._evaluate_one(pid))
                    active.add(t)
                    t.add_done_callback(active.discard)
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            for t in active:
                t.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            raise

    async def _evaluate_one(self, pid: str) -> None:
        try:
            await self._storage.batch_transition_by_ids(
                [pid],
                ProgramState.QUEUED.value,
                ProgramState.RUNNING.value,
            )
            await asyncio.sleep(self._delay)
            prog = await self._storage.get(pid)
            if prog is None:
                return
            prog.metrics.setdefault("fitness", 1.0)
            await self._storage.update(prog)
            await self._storage.batch_transition_by_ids(
                [pid],
                ProgramState.RUNNING.value,
                ProgramState.DONE.value,
            )
            self.flip_count[pid] += 1
        finally:
            self._claimed.discard(pid)


class _UniqueValueOperator(MutationOperator):
    """Generates distinct mutants — each call emits a unique entrypoint."""

    def __init__(self, start: int = 0) -> None:
        self._counter = start
        self._lock = asyncio.Lock()

    async def mutate_single(
        self,
        selected_parents: list[Program],
        memory_instructions: str | None = None,
    ) -> MutationSpec | None:
        async with self._lock:
            self._counter += 1
            n = self._counter
        return MutationSpec(
            code=f"def entrypoint():\n    return {n}",
            parents=selected_parents,
            name=f"resume-mutant-{n}",
        )


async def _seed_archive(storage: RedisProgramStorage, n: int) -> list[Program]:
    seeds: list[Program] = []
    for i in range(n):
        prog = Program(
            code=f"def entrypoint():\n    return -{i + 1}",
            state=ProgramState.DONE,
            metrics={"fitness": 0.5},
        )
        await storage.add(prog)
        seeds.append(prog)
    return seeds


def _build_engine(
    storage: RedisProgramStorage,
    strategy: _StubStrategy,
    operator: _UniqueValueOperator,
    *,
    max_in_flight: int,
    cap: int,
) -> SteadyStateEvolutionEngine:
    tracker = MagicMock()
    tracker.start = MagicMock()

    async def _stop() -> None:
        return None

    tracker.stop = _stop
    tracker.get_best_fitness = MagicMock(return_value=0.0)

    config = SteadyStateEngineConfig(
        loop_interval=0.001,
        max_elites_per_generation=max(1, max_in_flight),
        max_in_flight=max_in_flight,
        parent_selector=RandomParentSelector(num_parents=1),
        stopper=MaxMutantsStopper(cap),
    )
    return SteadyStateEvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=operator,
        config=config,
        writer=_make_null_writer(),
        metrics_tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
async def test_resume_after_kill_continues_from_snapshot() -> None:
    """Engine A runs to cap=5, is torn down, engine B picks up at the same
    storage with cap=10 and finishes the remaining mutants."""
    server = fakeredis.FakeServer()

    # ---- Engine A: cold-start run to cap=5 ----
    storage_a = _make_fakeredis_storage(server)
    seed_a = await _seed_archive(storage_a, 4)
    strategy_a = _StubStrategy(seed_a)
    operator_a = _UniqueValueOperator()

    engine_a = _build_engine(
        storage_a,
        strategy_a,
        operator_a,
        max_in_flight=2,
        cap=5,
    )
    dag_a = _TimedDag(storage_a)
    dag_a.start()
    try:
        await asyncio.wait_for(engine_a.run(), timeout=30.0)
    finally:
        await dag_a.stop()

    assert 5 <= engine_a.metrics.iteration <= 5 + 2, (
        f"engine A finished outside cap window: {engine_a.metrics.iteration}"
    )
    a_total_mutants = engine_a.metrics.iteration
    a_programs_processed = engine_a.metrics.programs_processed

    # Engine A snapshot survived the run.
    assert engine_a._snapshot.total_mutants == a_total_mutants
    assert engine_a._snapshot.programs_processed == a_programs_processed

    # ---- Simulate process death: close A, drop the instance ----
    await storage_a.close()
    del engine_a, strategy_a, operator_a, dag_a, storage_a

    # ---- Engine B: fresh instance against the same fakeredis server ----
    storage_b = _make_fakeredis_storage(server)

    # Rebuild the strategy from storage: read all DONE programs and seed
    # the in-memory archive so the engine can pick parents on the first
    # dispatch. In the real runner this happens through
    # ``EvolutionStrategy.restore_state``; here we replicate it minimally.
    done_ids = await storage_b.get_ids_by_status(ProgramState.DONE.value)
    done_programs = [
        p
        for p in await storage_b.mget(done_ids)
        if p is not None and p.state == ProgramState.DONE
    ]
    strategy_b = _StubStrategy(done_programs)
    operator_b = _UniqueValueOperator(start=a_total_mutants)

    engine_b = _build_engine(
        storage_b,
        strategy_b,
        operator_b,
        max_in_flight=2,
        cap=10,
    )

    # Restore counters from the snapshot persisted by engine A.
    await engine_b.restore_state()
    assert engine_b.metrics.iteration == a_total_mutants, (
        f"restore_state did not hydrate total_mutants: "
        f"got {engine_b.metrics.iteration}, expected {a_total_mutants}"
    )
    assert engine_b.metrics.programs_processed == a_programs_processed

    dag_b = _TimedDag(storage_b)
    dag_b.start()
    try:
        await asyncio.wait_for(engine_b.run(), timeout=30.0)
    finally:
        await dag_b.stop()
        await storage_b.close()

    # ---- Invariant: engine B reached cap=10 with bounded overshoot ----
    assert 10 <= engine_b.metrics.iteration <= 10 + 2, (
        f"engine B finished outside cap window: {engine_b.metrics.iteration}"
    )

    # ---- Invariant: no slot leak on engine B's exit ----
    assert engine_b._producer_sema._value == 2 and engine_b._buffer_sema._value == 2, (
        f"engine B sema leak: producer={engine_b._producer_sema._value} "
        f"buffer={engine_b._buffer_sema._value}"
    )
    assert not engine_b._in_flight, (
        f"engine B did not drain in_flight: {len(engine_b._in_flight)}"
    )

    # ---- Invariant: progress is strictly forward across the resume ----
    assert engine_b.metrics.iteration > a_total_mutants, (
        "resumed engine did not produce additional mutants"
    )
    assert engine_b.metrics.programs_processed >= a_programs_processed, (
        f"programs_processed regressed across resume: "
        f"{a_programs_processed} -> {engine_b.metrics.programs_processed}"
    )

    # ---- Invariant: snapshot is consistent with metrics at end ----
    assert engine_b._snapshot.total_mutants == engine_b.metrics.iteration
    assert engine_b._snapshot.programs_processed == engine_b.metrics.programs_processed


__all__: list[str] = []
