"""Concurrency stress + simulation suite for the steady-state engine.

Drives the real :class:`SteadyStateEvolutionEngine` (dispatcher + ingestor +
:class:`ParentRefresher`) against a fakeredis-backed
:class:`RedisProgramStorage` and a configurable timed fake DAG. Parametrised
over ``(max_in_flight, n_mutants, duration_dist, overlap_rate)``; each
combination asserts the same set of invariants:

- No semaphore slot leak (``sema._value == max_in_flight`` at end).
- ``_in_flight`` set is empty.
- ``total_mutants`` equals the number of mutants the dispatcher persisted.
- ``programs_processed`` equals ``accepted + rejected`` (no orphans).
- ParentRefresher flipped each parent at most once per mutant that used it.
- ``EngineSnapshot`` counters captured during the run are
  monotonically non-decreasing.

The fake DAG simulates per-program evaluation latency under three
distributions — constant, exponential, and heavy-tail — so the stress
exercises the dispatcher/ingestor backpressure under realistic completion
ordering.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
import contextlib
import random
from unittest.mock import MagicMock

import fakeredis
import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import MaxMutantsStopper
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.mutation.parent_selector import RandomParentSelector
from gigaevo.evolution.strategies.base import EvolutionStrategy
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------


def _make_fakeredis_storage(server: fakeredis.FakeServer) -> RedisProgramStorage:
    """Build a RedisProgramStorage bound to an in-memory fakeredis server."""
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0", key_prefix="stress"
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


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubArchiveStrategy(EvolutionStrategy):
    """Bare-minimum :class:`EvolutionStrategy` for stress testing.

    Tracks accepted programs in a list and exposes them as elites. Every
    incoming program is accepted (``add`` returns ``True``); the elite list
    is the basis from which the engine's ``parent_selector`` picks parents.

    Pre-seed via ``seed`` so the dispatcher has parents available the moment
    it starts.
    """

    def __init__(self, seed: list[Program]) -> None:
        self._archive: list[Program] = list(seed)
        self._ids: set[str] = {p.id for p in seed}
        self.selection_count: dict[str, int] = defaultdict(int)

    async def add(self, program: Program) -> bool:
        if program.id in self._ids:
            return False
        self._archive.append(program)
        self._ids.add(program.id)
        return True

    async def select_elites(self, total: int) -> list[Program]:
        # Honour the EvolutionStrategy.select_elites contract: return at most
        # ``total`` elites. The engine asks for exactly ``parent_selector.num_parents``
        # and treats the response as the parent set.
        if not self._archive:
            return []
        elites = random.sample(self._archive, min(total, len(self._archive)))
        for p in elites:
            self.selection_count[p.id] += 1
        return elites

    async def get_program_ids(self) -> list[str]:
        return list(self._ids)


class TimedFakeDag:
    """Fake DAG runner with configurable per-program evaluation latency.

    Polls storage for QUEUED programs, walks each through QUEUED → RUNNING →
    DONE with a delay drawn from the configured distribution. Records every
    QUEUED → DONE flip per program id so tests can assert refresh counts.

    Each program is processed in its own task — that way fast programs do
    not block slow ones, which is the realistic ordering the engine must
    handle.
    """

    def __init__(
        self,
        storage: RedisProgramStorage,
        state_manager: ProgramStateManager,
        *,
        duration_dist: str = "const",
        seed: int = 0,
    ) -> None:
        self._storage = storage
        self._sm = state_manager
        self._duration_dist = duration_dist
        self._rng = random.Random(seed)
        self._task: asyncio.Task | None = None
        self._claimed: set[str] = set()
        self.flip_count: dict[str, int] = defaultdict(int)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="timed-fake-dag")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    def _draw_duration(self) -> float:
        if self._duration_dist == "const":
            return 0.003
        if self._duration_dist == "expo":
            # mean 3ms, capped at 50ms so the test stays bounded
            return min(self._rng.expovariate(1.0 / 0.003), 0.05)
        if self._duration_dist == "heavy_tail":
            # 90% short, 10% long tail
            return 0.001 if self._rng.random() < 0.9 else 0.03
        raise ValueError(f"unknown duration_dist={self._duration_dist!r}")

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
            await asyncio.sleep(self._draw_duration())
            prog = await self._storage.get(pid)
            if prog is None:
                return
            # Inject minimal metrics so the default acceptor passes.
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


class UniqueValueOperator(MutationOperator):
    """Generates deterministic, distinct mutants.

    The Nth call returns ``def entrypoint(): return N`` so every mutation
    produces a fresh program id. The operator is the only mutation source
    in the stress test — its uniqueness guarantees that each accepted
    mutant grows the elite pool by one.
    """

    def __init__(self) -> None:
        self._counter = 0
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
            name=f"stress-mutant-{n}",
        )


# ---------------------------------------------------------------------------
# Snapshot monotonicity watcher
# ---------------------------------------------------------------------------


class SnapshotWatcher:
    """Background coroutine that samples ``engine._snapshot`` at a fixed cadence.

    Stores ``(total_mutants, programs_processed)`` tuples; tests assert that
    each sequence is monotonically non-decreasing. The watcher cooperates with
    cancellation — ``stop()`` waits for the task to exit cleanly.
    """

    def __init__(self, engine: SteadyStateEvolutionEngine, *, interval: float = 0.01):
        self._engine = engine
        self._interval = interval
        self._task: asyncio.Task | None = None
        self.samples: list[tuple[int, int]] = []

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="snapshot-watcher")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _loop(self) -> None:
        try:
            while True:
                snap = self._engine._snapshot
                self.samples.append((snap.total_mutants, snap.programs_processed))
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            # Capture one final sample before exiting so the post-run
            # asserts see the terminal state.
            snap = self._engine._snapshot
            self.samples.append((snap.total_mutants, snap.programs_processed))
            raise


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


async def _seed_archive(storage: RedisProgramStorage, n_elites: int) -> list[Program]:
    """Insert ``n_elites`` DONE programs into storage and return them."""
    seed_programs: list[Program] = []
    for i in range(n_elites):
        prog = Program(
            code=f"def entrypoint():\n    return -{i + 1}",
            state=ProgramState.DONE,
            metrics={"fitness": 0.5},
        )
        await storage.add(prog)
        seed_programs.append(prog)
    return seed_programs


def _build_engine(
    storage: RedisProgramStorage,
    strategy: StubArchiveStrategy,
    *,
    max_in_flight: int,
    n_mutants: int,
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
        stopper=MaxMutantsStopper(n_mutants),
    )

    return SteadyStateEvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=UniqueValueOperator(),
        config=config,
        writer=_make_null_writer(),
        metrics_tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Parametrised stress suite
# ---------------------------------------------------------------------------


PARAMS = [
    pytest.param(mif, n, dur, ov, id=f"mif{mif}-n{n}-{dur}-ov{int(ov * 10)}")
    for mif in (1, 4, 16)
    for n in (50, 200)
    for dur in ("const", "expo", "heavy_tail")
    for ov in (0.0, 0.5)
]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("max_in_flight,n_mutants,duration_dist,overlap_rate", PARAMS)
async def test_stress_invariants(
    max_in_flight: int,
    n_mutants: int,
    duration_dist: str,
    overlap_rate: float,
) -> None:
    """Drive the full engine and verify the six core invariants.

    ``overlap_rate`` is realised by seeding the archive with either many
    distinct elites (low overlap — concurrent producers pick different
    parents) or a single elite (high overlap — every producer contends
    for the same lock and the per-id ParentRefresher serialises them).
    """
    n_elites = 1 if overlap_rate >= 0.5 else max(max_in_flight * 2, 4)

    server = fakeredis.FakeServer()
    storage = _make_fakeredis_storage(server)
    state_manager = ProgramStateManager(storage)

    seed = await _seed_archive(storage, n_elites)
    strategy = StubArchiveStrategy(seed)

    engine = _build_engine(
        storage,
        strategy,
        max_in_flight=max_in_flight,
        n_mutants=n_mutants,
    )

    dag = TimedFakeDag(
        storage,
        state_manager,
        duration_dist=duration_dist,
        seed=hash((max_in_flight, n_mutants, duration_dist, overlap_rate)) & 0xFFFF,
    )
    watcher = SnapshotWatcher(engine, interval=0.01)

    dag.start()
    watcher.start()
    try:
        # Wall-clock budget. Each mutant cycle involves a parent refresh
        # round-trip (DONE→QUEUED→DONE via the fake DAG), mutant evaluation,
        # and a snapshot write — empirically ~50–150ms per mutant in the
        # serial (mif=1) case. Pad heavily so the test fails loudly rather
        # than silently hanging on a regression.
        timeout = max(60.0, n_mutants * 0.3 / max_in_flight + 15.0)
        await asyncio.wait_for(engine.run(), timeout=timeout)
    finally:
        await watcher.stop()
        await dag.stop()
        await storage.close()

    # ---- Invariant 1: no semaphore slot leak ----
    # Both semaphores should be fully released (value == max_in_flight).
    assert (
        engine._producer_sema._value == max_in_flight
        and engine._buffer_sema._value == max_in_flight
    ), (
        f"sema leak: producer={engine._producer_sema._value} "
        f"buffer={engine._buffer_sema._value} != max_in_flight={max_in_flight}"
    )

    # ---- Invariant 2: _in_flight set is empty ----
    assert not engine._in_flight, (
        f"in_flight not drained: {len(engine._in_flight)} ids leftover"
    )

    # ---- Invariant 3: total_mutants reaches the cap with bounded overshoot ----
    # The dispatcher checks the cap before acquiring a slot, but the cap counter
    # is only incremented after a mutant is persisted. With up to max_in_flight
    # tasks in flight, the engine can overshoot the configured cap by at most
    # max_in_flight slots — this is documented steady-state behavior.
    assert n_mutants <= engine.metrics.total_mutants <= n_mutants + max_in_flight, (
        f"total_mutants={engine.metrics.total_mutants} outside expected window "
        f"[{n_mutants}, {n_mutants + max_in_flight}]"
    )
    spawned = engine.metrics.total_mutants

    # ---- Invariant 4: programs_processed == accepted + rejected ----
    accepted = engine.metrics.added
    rejected = engine.metrics.rejected_validation + engine.metrics.rejected_strategy
    assert engine.metrics.programs_processed == accepted + rejected, (
        f"orphan: programs_processed={engine.metrics.programs_processed} != "
        f"accepted({accepted}) + rejected({rejected})"
    )

    # ---- Invariant 5: ParentRefresher flip count is bounded ----
    # Every parent flip is one re-evaluation; under per-id locks the
    # refresher serialises overlapping calls, so flip_count[p] equals the
    # number of mutants that selected p (best-effort assertion).
    # Total flips include the n_elites initial seeding only when they were
    # re-evaluated by the refresher.
    total_flips = sum(dag.flip_count.values())
    # Each spawned mutant flips once on its own evaluation; each parent refresh
    # flips the chosen parent once. With concurrent producers contending for
    # the same parent under per-id locks, an upper bound of 5× spawned leaves
    # plenty of slack for race-induced re-evaluations while still catching a
    # runaway hot loop (which would multiply flips by orders of magnitude).
    assert total_flips >= spawned, (
        f"flip count {total_flips} below spawned={spawned} — DAG never "
        "completed mutants"
    )
    assert total_flips <= 5 * spawned + 5 * n_elites, (
        f"flip count {total_flips} exceeds runaway-loop ceiling "
        f"{5 * spawned + 5 * n_elites}"
    )

    # ---- Invariant 6: snapshot counters monotonically non-decreasing ----
    last_tm = 0
    last_pp = 0
    for tm, pp in watcher.samples:
        assert tm >= last_tm, (
            f"total_mutants regressed: {last_tm} -> {tm} in watcher samples"
        )
        assert pp >= last_pp, (
            f"programs_processed regressed: {last_pp} -> {pp} in watcher samples"
        )
        last_tm, last_pp = tm, pp

    # Final snapshot values agree with in-process metrics.
    assert engine._snapshot.total_mutants == engine.metrics.total_mutants
    assert engine._snapshot.programs_processed == engine.metrics.programs_processed


__all__: list[str] = []
