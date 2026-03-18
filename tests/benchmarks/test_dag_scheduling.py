"""Benchmark: DagRunner throughput without engine overhead.

Measures raw DAG scheduling and execution performance by pushing
heavy programs through the DagRunner directly.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import fakeredis
import pytest

from gigaevo.programs.program_state import ProgramState
from gigaevo.runner.dag_runner import DagRunner, DagRunnerConfig
from tests.benchmarks.conftest import (
    cleanup_storage,
    make_benchmark_blueprint,
    make_heavy_program,
    make_storage,
)

pytestmark = pytest.mark.benchmark

NUM_PROGRAMS = 50


def _make_null_writer() -> MagicMock:
    writer = MagicMock()
    writer.bind.return_value = writer
    return writer


async def _wait_for_done(storage, target_count, timeout_iters=6000):
    """Poll until target_count programs are DONE+DISCARDED."""
    done_count = 0
    discarded_count = 0
    for _ in range(timeout_iters):
        done_count = await storage.count_by_status(ProgramState.DONE.value)
        discarded_count = await storage.count_by_status(ProgramState.DISCARDED.value)
        if done_count + discarded_count >= target_count:
            break
        await asyncio.sleep(0.01)
    return done_count, discarded_count


@pytest.fixture(params=[1, 4, 8, 16])
def concurrency(request):
    return request.param


class TestDagThroughput:
    """Programs/sec through DagRunner at various concurrency levels."""

    async def test_dag_throughput(
        self, concurrency: int, redis_url: str | None
    ) -> None:
        server = None if redis_url else fakeredis.FakeServer()
        storage = make_storage(server=server, redis_url=redis_url)
        blueprint = make_benchmark_blueprint()
        writer = _make_null_writer()

        dag_runner = DagRunner(
            storage=storage,
            dag_blueprint=blueprint,
            config=DagRunnerConfig(
                poll_interval=0.01,
                max_concurrent_dags=concurrency,
                dag_timeout=30.0,
            ),
            writer=writer,
        )

        # Create QUEUED heavy programs
        for i in range(NUM_PROGRAMS):
            p = make_heavy_program(float(i), float(i % 10))
            p.state = ProgramState.QUEUED
            # Clear stage_results so the DAG actually runs stages
            p.stage_results = {}
            await storage.add(p)

        dag_runner.start()
        start = time.perf_counter()

        done_count, discarded_count = await _wait_for_done(storage, NUM_PROGRAMS)
        elapsed_s = time.perf_counter() - start

        total_processed = done_count + discarded_count
        await dag_runner.stop()

        assert total_processed >= NUM_PROGRAMS, (
            f"Only {done_count} DONE + {discarded_count} DISCARDED "
            f"out of {NUM_PROGRAMS}"
        )

        progs_per_s = NUM_PROGRAMS / elapsed_s
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: dag_throughput concurrency={concurrency} ({backend}): "
            f"{progs_per_s:.1f} prog/s ({elapsed_s:.3f}s for {NUM_PROGRAMS} programs)"
        )
        if redis_url:
            cleanup_s = make_storage(redis_url=redis_url)
            await cleanup_storage(cleanup_s)


class TestSchedulingOverhead:
    """Per-program DAG scheduling overhead."""

    async def test_scheduling_overhead(self, redis_url: str | None) -> None:
        writer = _make_null_writer()

        # --- Phase 1: measure single-DAG time ---
        server1 = None if redis_url else fakeredis.FakeServer()
        storage1 = make_storage(server=server1, redis_url=redis_url)
        dag_runner_1 = DagRunner(
            storage=storage1,
            dag_blueprint=make_benchmark_blueprint(),
            config=DagRunnerConfig(
                poll_interval=0.01,
                max_concurrent_dags=1,
                dag_timeout=30.0,
            ),
            writer=writer,
        )

        p = make_heavy_program(1.0, 1.0)
        p.state = ProgramState.QUEUED
        p.stage_results = {}
        await storage1.add(p)
        dag_runner_1.start()

        start = time.perf_counter()
        await _wait_for_done(storage1, 1, timeout_iters=3000)
        single_dag_time = time.perf_counter() - start
        await dag_runner_1.stop()

        # --- Phase 2: measure N-program time with concurrency=8 ---
        server2 = None if redis_url else fakeredis.FakeServer()
        storage2 = make_storage(server=server2, redis_url=redis_url)
        dag_runner_n = DagRunner(
            storage=storage2,
            dag_blueprint=make_benchmark_blueprint(),
            config=DagRunnerConfig(
                poll_interval=0.01,
                max_concurrent_dags=8,
                dag_timeout=30.0,
            ),
            writer=writer,
        )

        n = 30
        for i in range(n):
            pp = make_heavy_program(float(i + 10), float(i % 10))
            pp.state = ProgramState.QUEUED
            pp.stage_results = {}
            await storage2.add(pp)

        dag_runner_n.start()
        start = time.perf_counter()
        await _wait_for_done(storage2, n)
        total_time = time.perf_counter() - start
        await dag_runner_n.stop()

        overhead_per_prog_ms = (
            (total_time - n * single_dag_time) / n * 1000
            if total_time > n * single_dag_time
            else 0.0
        )
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: scheduling_overhead ({backend}): "
            f"{overhead_per_prog_ms:.1f}ms/program "
            f"(single={single_dag_time * 1000:.1f}ms, "
            f"total={total_time * 1000:.0f}ms for {n} programs)"
        )
        if redis_url:
            for s in [
                make_storage(redis_url=redis_url),
                make_storage(redis_url=redis_url),
            ]:
                await cleanup_storage(s)
