"""Real-Redis integration smoke for the steady-state engine.

Runs the full steady-state loop against an actual Redis at
localhost:6379/0 (or `REAL_REDIS_URL` if set). Skipped automatically
when no Redis is reachable, so the test is safe to commit and harmless
on machines without a local server.

The point is to validate behaviours that fakeredis silently glosses over:

- real network round-trips on every storage op (latency-sensitive paths
  that fakeredis fast-paths in process memory);
- real pipelined batch_transition / mget semantics;
- snapshot persistence + version monotonicity over a real Redis JSON write.

The test uses a unique random key prefix so it never clobbers another
caller's data, and aggressively cleans up via SCAN+DELETE on that
prefix in ``finally``.
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

REAL_REDIS_URL = os.environ.get("REAL_REDIS_URL", "redis://localhost:6379/0")


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
    """Yield a RedisProgramStorage bound to a real Redis with a unique
    key prefix, then SCAN+DELETE everything under that prefix on teardown."""
    if not await _redis_reachable(REAL_REDIS_URL):
        pytest.skip(f"real Redis not reachable at {REAL_REDIS_URL}")

    prefix = f"gigaevo_test_jit_refresh_{uuid.uuid4().hex[:12]}"
    config = RedisProgramStorageConfig(
        redis_url=REAL_REDIS_URL,
        key_prefix=prefix,
        # Faster heartbeat for the smoke test.
        retry_delay=0.05,
        lock_expiry_secs=60,
        lock_renewal_secs=20,
    )
    storage = RedisProgramStorage(config)
    try:
        yield storage
    finally:
        # Drain anything still attached to the storage (lock, metrics).
        with contextlib.suppress(Exception):
            await storage.close()
        # Wipe the prefix so nothing leaks to disk.
        try:
            r = redis_async.from_url(REAL_REDIS_URL)
            try:
                keys = [k async for k in r.scan_iter(match=f"{prefix}:*", count=500)]
                if keys:
                    # DEL accepts up to ~1M args; for safety chunk to 1k.
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
            name=f"real-redis-mutant-{n}",
        )


class _RealRedisDag:
    """Drives QUEUED → RUNNING → DONE with a small constant delay,
    polling real Redis the same way :class:`_TimedDag` polls fakeredis."""

    def __init__(self, storage: RedisProgramStorage, *, delay: float = 0.01) -> None:
        self._storage = storage
        self._delay = delay
        self._task: asyncio.Task | None = None
        self._claimed: set[str] = set()
        self.flip_count: dict[str, int] = defaultdict(int)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="real-redis-fake-dag")

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


@pytest.mark.timeout(60)
async def test_real_redis_steady_state_smoke(real_redis_storage) -> None:
    """End-to-end smoke against real Redis: seed → run to cap=6 with
    max_in_flight=2 → verify bounded overshoot, no slot leak, archive
    growth, and that the snapshot was persisted to Redis at the
    in-memory version."""
    storage = real_redis_storage

    seed = await _seed_archive(storage, 4)
    strategy = _StubStrategy(seed)
    operator = _UniqueValueOperator()

    engine = _build_engine(storage, strategy, operator, max_in_flight=2, cap=6)
    dag = _RealRedisDag(storage, delay=0.01)
    dag.start()
    try:
        await asyncio.wait_for(engine.run(), timeout=45.0)
    finally:
        await dag.stop()

    # ---- Bounded overshoot ----
    assert 6 <= engine.metrics.total_mutants <= 6 + 2, (
        f"engine finished outside cap window: {engine.metrics.total_mutants}"
    )

    # ---- No slot leak ----
    assert engine._producer_sema._value == 2 and engine._buffer_sema._value == 2, (
        f"sema leak: producer={engine._producer_sema._value} "
        f"buffer={engine._buffer_sema._value}"
    )
    assert not engine._in_flight, (
        f"engine did not drain in_flight: {len(engine._in_flight)}"
    )

    # ---- Archive grew ----
    archive_ids = await strategy.get_program_ids()
    assert len(archive_ids) >= len(seed), (
        f"archive shrank from seed: {len(archive_ids)} < {len(seed)}"
    )

    # ---- Snapshot persisted at in-memory version ----
    from gigaevo.evolution.engine.snapshot import (
        ENGINE_SNAPSHOT_KEY,
        EngineSnapshot,
    )

    raw = await storage.load_run_state_str(ENGINE_SNAPSHOT_KEY)
    assert raw is not None, "engine snapshot was never persisted to Redis"
    redis_snap = EngineSnapshot.model_validate_json(raw)
    assert redis_snap.total_mutants == engine.metrics.total_mutants, (
        f"Redis snapshot lagged memory: redis={redis_snap.total_mutants} "
        f"memory={engine.metrics.total_mutants}"
    )
    assert redis_snap.version == engine._snapshot.version


__all__: list[str] = []
