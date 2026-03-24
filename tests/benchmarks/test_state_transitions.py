"""Benchmark: ProgramStateManager transition throughput.

State transitions (QUEUED -> RUNNING -> DONE) are the inner loop of the
DagRunner. Each transition acquires a per-program lock and persists to
Redis. This benchmark measures sequential and concurrent throughput.
"""

from __future__ import annotations

import asyncio

import fakeredis
import pytest

from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.benchmarks.conftest import (
    BenchmarkTimer,
    cleanup_storage,
    make_heavy_program,
    make_storage,
)

pytestmark = pytest.mark.benchmark

NUM_PROGRAMS = 100


def _make_queued_programs(n: int) -> list[Program]:
    return [
        Program(code=f"def run_code(): return {i}", state=ProgramState.QUEUED)
        for i in range(n)
    ]


class TestSequentialTransitions:
    """QUEUED -> RUNNING -> DONE for N programs, one at a time."""

    async def test_sequential_full_lifecycle(self, redis_url: str | None) -> None:
        server = None if redis_url else fakeredis.FakeServer()
        storage = make_storage(server=server, redis_url=redis_url)
        sm = ProgramStateManager(storage)

        programs = _make_queued_programs(NUM_PROGRAMS)
        for p in programs:
            await storage.add(p)

        with BenchmarkTimer() as t:
            for p in programs:
                await sm.set_program_state(p, ProgramState.RUNNING)
                await sm.set_program_state(p, ProgramState.DONE)

        transitions = NUM_PROGRAMS * 2
        rate = transitions / (t.elapsed_ms / 1000) if t.elapsed_ms > 0 else 0
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: sequential_transitions N={NUM_PROGRAMS} ({backend}): "
            f"{t.elapsed_ms:.1f}ms ({rate:.0f} transitions/s)"
        )
        await cleanup_storage(storage)


class TestConcurrentTransitions:
    """Parallel QUEUED -> RUNNING -> DONE (simulates DagRunner scheduling)."""

    @pytest.fixture(params=[4, 8, 16])
    def concurrency(self, request):
        return request.param

    async def test_concurrent_lifecycle(
        self, concurrency: int, redis_url: str | None
    ) -> None:
        server = None if redis_url else fakeredis.FakeServer()
        storage = make_storage(server=server, redis_url=redis_url)
        sm = ProgramStateManager(storage)

        programs = _make_queued_programs(NUM_PROGRAMS)
        for p in programs:
            await storage.add(p)

        async def lifecycle(p: Program) -> None:
            await sm.set_program_state(p, ProgramState.RUNNING)
            await sm.set_program_state(p, ProgramState.DONE)

        # Process in batches of `concurrency`
        with BenchmarkTimer() as t:
            for start in range(0, NUM_PROGRAMS, concurrency):
                batch = programs[start : start + concurrency]
                await asyncio.gather(*[lifecycle(p) for p in batch])

        transitions = NUM_PROGRAMS * 2
        rate = transitions / (t.elapsed_ms / 1000) if t.elapsed_ms > 0 else 0
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: concurrent_transitions concurrency={concurrency} ({backend}): "
            f"{t.elapsed_ms:.1f}ms ({rate:.0f} transitions/s)"
        )
        await cleanup_storage(storage)


class TestCountByStatusUnderLoad:
    """count_by_status while programs are in various states."""

    async def test_count_under_load(self, redis_url: str | None) -> None:
        server = None if redis_url else fakeredis.FakeServer()
        storage = make_storage(server=server, redis_url=redis_url)
        sm = ProgramStateManager(storage)

        # Create a mix of states
        programs = _make_queued_programs(200)
        for p in programs:
            await storage.add(p)
        # Transition some to RUNNING, some to DONE
        for i, p in enumerate(programs):
            if i % 3 == 0:
                await sm.set_program_state(p, ProgramState.RUNNING)
            elif i % 3 == 1:
                await sm.set_program_state(p, ProgramState.RUNNING)
                await sm.set_program_state(p, ProgramState.DONE)

        k = 200
        with BenchmarkTimer() as t:
            for _ in range(k):
                await storage.count_by_status("queued")
                await storage.count_by_status("running")
                await storage.count_by_status("done")

        avg_ms = t.elapsed_ms / k
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: count_by_status x3 ({backend}): "
            f"{avg_ms:.3f}ms/round ({t.elapsed_ms:.0f}ms for {k} rounds)"
        )
        await cleanup_storage(storage)


class TestBatchTransitionByIds:
    """Compare batch_transition_by_ids (raw JSON patch) vs batch_transition_state."""

    @pytest.fixture(params=[100, 500, 2000])
    def batch_size(self, request):
        return request.param

    async def test_batch_by_ids_heavy(
        self, batch_size: int, redis_url: str | None
    ) -> None:
        """batch_transition_by_ids on heavy programs (the refresh path)."""
        server = None if redis_url else fakeredis.FakeServer()
        storage = make_storage(server=server, redis_url=redis_url)

        ids = []
        for i in range(batch_size):
            p = make_heavy_program(float(i), float(i % 10))
            p.state = ProgramState.DONE
            await storage.add(p)
            ids.append(p.id)

        with BenchmarkTimer() as t:
            count = await storage.batch_transition_by_ids(
                ids,
                ProgramState.DONE.value,
                ProgramState.QUEUED.value,
            )

        assert count == batch_size
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: batch_by_ids_heavy N={batch_size} ({backend}): "
            f"{t.elapsed_ms:.1f}ms"
        )
        await cleanup_storage(storage)

    async def test_batch_state_heavy(
        self, batch_size: int, redis_url: str | None
    ) -> None:
        """batch_transition_state on heavy programs (the old path)."""
        server = None if redis_url else fakeredis.FakeServer()
        storage = make_storage(server=server, redis_url=redis_url)

        programs = []
        for i in range(batch_size):
            p = make_heavy_program(float(i), float(i % 10))
            p.state = ProgramState.DONE
            await storage.add(p)
            programs.append(p)

        with BenchmarkTimer() as t:
            count = await storage.batch_transition_state(
                programs,
                ProgramState.DONE.value,
                ProgramState.QUEUED.value,
            )

        assert count == batch_size
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: batch_state_heavy N={batch_size} ({backend}): "
            f"{t.elapsed_ms:.1f}ms"
        )
        await cleanup_storage(storage)
