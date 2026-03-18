"""Benchmark: WorkerPool contention with real subprocesses.

Marked @slow — skipped by default in benchmark runs. Run with --full flag
or `pytest tests/benchmarks/test_worker_pool.py -v -s`.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from gigaevo.programs.stages.python_executors.wrapper import (
    WorkerPool,
    run_exec_runner,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

_CODE = "def run_code():\n    return 1 + 1"


class TestPoolSerialThroughput:
    """Sequential run_exec_runner calls/sec."""

    async def test_pool_serial_throughput(self) -> None:
        pool = WorkerPool(max_workers=2)
        n = 20
        start = time.perf_counter()
        for _ in range(n):
            return_value, stdout, stderr = await run_exec_runner(
                code=_CODE,
                function_name="run_code",
                timeout=10,
                pool=pool,
            )
            assert return_value == 2
        elapsed_s = time.perf_counter() - start

        calls_per_s = n / elapsed_s
        print(
            f"BENCHMARK: pool_serial_throughput: "
            f"{calls_per_s:.1f} calls/s ({elapsed_s:.3f}s for {n} calls)"
        )
        await pool.shutdown()


class TestPoolContention:
    """N callers competing for M workers simultaneously."""

    @pytest.fixture(
        params=[
            (2, 4),  # 2 workers, 4 callers
            (4, 8),  # 4 workers, 8 callers
            (4, 16),  # 4 workers, 16 callers (heavy contention)
        ]
    )
    def workers_callers(self, request):
        return request.param

    async def test_pool_contention(self, workers_callers: tuple[int, int]) -> None:
        workers, callers = workers_callers
        pool = WorkerPool(max_workers=workers)

        async def _call():
            return await run_exec_runner(
                code=_CODE,
                function_name="run_code",
                timeout=10,
                pool=pool,
            )

        start = time.perf_counter()
        results = await asyncio.gather(*[_call() for _ in range(callers)])
        elapsed_s = time.perf_counter() - start

        assert len(results) == callers
        for return_value, stdout, stderr in results:
            assert return_value == 2

        calls_per_s = callers / elapsed_s
        print(
            f"BENCHMARK: pool_contention workers={workers} callers={callers}: "
            f"{calls_per_s:.1f} calls/s ({elapsed_s:.3f}s)"
        )
        await pool.shutdown()
