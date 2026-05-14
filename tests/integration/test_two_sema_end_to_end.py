"""End-to-end real-Redis smoke for the two-sema steady-state pipeline.

Validates the two-sema model under the full DAG + ingestor stack — the
unit/invariant tests in ``tests/evolution/`` exercise individual
components with fake-engine surfaces; this one wires the real thing.

Runs the steady-state engine with ``max_in_flight=3`` and a cap of 30
mutants against a real Redis at ``localhost:6379`` DB 15 (the tests-only
DB). Asserts the drain contract: when ``engine.run()`` returns, both
``_producer_sema`` and ``_buffer_sema`` are back at full capacity (3),
``_in_flight`` is empty, and ``total_mutants`` reached the cap.

Skipped automatically if Redis DB 15 is not reachable, so the test is
safe to commit and harmless on machines without a local server.

Mirrors the helper pattern from
``tests/integration/test_engine_real_redis.py`` with a distinct random
key prefix so two integration tests in the same run never collide.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
import contextlib
import os
from unittest.mock import MagicMock
import uuid

import pytest
import redis.asyncio as redis_async

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

# Honor REAL_REDIS_URL as a full-URL override first (matching the
# sibling test_engine_real_redis.py), falling back to host/port/db
# construction so CI configs can use either convention.
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = 15  # tests-only DB
REAL_REDIS_URL = os.environ.get(
    "REAL_REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
)


async def _redis_reachable(url: str) -> bool:
    try:
        r = redis_async.from_url(url, socket_connect_timeout=1)
        try:
            await r.ping()
        finally:
            await r.aclose()
        return True
    except Exception:
        return False


@pytest.fixture
async def real_redis_storage():
    """Yield a RedisProgramStorage bound to a real Redis DB 15 with a
    unique key prefix, then SCAN+DELETE everything under that prefix on
    teardown."""
    if not await _redis_reachable(REAL_REDIS_URL):
        pytest.skip(f"real Redis not reachable at {REAL_REDIS_URL}")

    # Distinct prefix from test_engine_real_redis.py so concurrent runs
    # cannot collide.
    prefix = f"gigaevo_test_two_sema_e2e_{uuid.uuid4().hex[:12]}"
    config = RedisProgramStorageConfig(
        redis_url=REAL_REDIS_URL,
        key_prefix=prefix,
        retry_delay=0.05,
        lock_expiry_secs=60,
        lock_renewal_secs=20,
    )
    storage = RedisProgramStorage(config)
    try:
        yield storage
    finally:
        with contextlib.suppress(Exception):
            await storage.close()
        try:
            r = redis_async.from_url(REAL_REDIS_URL)
            try:
                keys = [k async for k in r.scan_iter(match=f"{prefix}:*", count=500)]
                if keys:
                    for i in range(0, len(keys), 1000):
                        await r.delete(*keys[i : i + 1000])
            finally:
                await r.aclose()
        except Exception:
            # Cleanup failures are noisy but non-fatal for the suite.
            pass


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


class _UniqueValueOperator(MutationOperator):
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
            name=f"two-sema-mutant-{n}",
        )


class _RealRedisDag:
    """Drives QUEUED → RUNNING → DONE with a small constant delay,
    polling real Redis to mimic the production DAG worker."""

    def __init__(self, storage: RedisProgramStorage, *, delay: float = 0.01) -> None:
        self._storage = storage
        self._delay = delay
        self._task: asyncio.Task | None = None
        self._claimed: set[str] = set()
        self.flip_count: dict[str, int] = defaultdict(int)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="two-sema-fake-dag")

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
                await asyncio.sleep(0.005)
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
        loop_interval=0.005,
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


@pytest.mark.timeout(90)
async def test_two_sema_pipeline_drains_cleanly(real_redis_storage) -> None:
    """End-to-end drain contract: after run() completes, both producer
    and buffer semaphores return to full capacity (max_in_flight=3),
    ``_in_flight`` is empty, and the engine reached the cap of 30
    mutants."""
    storage = real_redis_storage

    seed = await _seed_archive(storage, 4)
    strategy = _StubStrategy(seed)
    operator = _UniqueValueOperator()

    engine = _build_engine(storage, strategy, operator, max_in_flight=3, cap=30)
    dag = _RealRedisDag(storage, delay=0.01)
    dag.start()
    try:
        await asyncio.wait_for(engine.run(), timeout=75.0)
    finally:
        await dag.stop()

    # ---- Drain contract ----
    assert not engine._in_flight, (
        f"engine did not drain in_flight: {len(engine._in_flight)}"
    )
    assert engine._producer_sema._value == 3, (
        f"producer_sema leaked: value={engine._producer_sema._value} (want 3)"
    )
    assert engine._buffer_sema._value == 3, (
        f"buffer_sema leaked: value={engine._buffer_sema._value} (want 3)"
    )
    # Bounded overshoot: cap=30, max_in_flight=3 ⇒ at most 30 + 3 mutants
    # can complete before the stopper drains the pipeline. An unbounded
    # overshoot would silently mask a stopper/drain-logic regression.
    assert 30 <= engine.metrics.total_mutants <= 30 + 3, (
        f"engine finished outside cap window: total_mutants="
        f"{engine.metrics.total_mutants} (expected 30..33)"
    )


__all__: list[str] = []
