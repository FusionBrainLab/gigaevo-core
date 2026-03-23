"""Benchmark: Realistic code sizes and payloads.

Real evolved programs are 1-3KB of Python, not the 30-byte
``def run_code(): return 42`` used in other benchmarks. Code size
affects serialization, Redis round-trips, and AST complexity analysis.
"""

from __future__ import annotations

import fakeredis
import pytest

from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.complexity import compute_numerical_complexity
from tests.benchmarks.conftest import (
    BenchmarkTimer,
    cleanup_storage,
    make_realistic_code,
    make_storage,
)

pytestmark = pytest.mark.benchmark


@pytest.fixture(params=[100, 500, 1000, 3000, 5000])
def code_size(request):
    """Target code size in bytes."""
    return request.param


class TestSerializationScaling:
    """to_dict() / from_dict() cost vs code size."""

    def test_serialization_roundtrip(self, code_size: int) -> None:
        code = make_realistic_code(code_size)
        p = Program(
            code=code,
            state=ProgramState.DONE,
            metrics={"fitness": 0.85, "is_valid": 1.0, "complexity": 42.0},
        )

        k = 500
        with BenchmarkTimer() as t_ser:
            for _ in range(k):
                d = p.to_dict()

        with BenchmarkTimer() as t_de:
            for _ in range(k):
                Program.from_dict(d)

        ser_avg = t_ser.elapsed_ms / k
        de_avg = t_de.elapsed_ms / k
        print(
            f"BENCHMARK: serialization code={code_size}B: "
            f"to_dict={ser_avg:.3f}ms from_dict={de_avg:.3f}ms"
        )


class TestStorageWithLargeCode:
    """Redis add/get round-trip with realistic code sizes."""

    async def test_storage_roundtrip(
        self, code_size: int, redis_url: str | None
    ) -> None:
        server = None if redis_url else fakeredis.FakeServer()
        storage = make_storage(server=server, redis_url=redis_url)
        code = make_realistic_code(code_size)

        programs = []
        k = 50
        with BenchmarkTimer() as t_add:
            for i in range(k):
                p = Program(
                    code=code,
                    state=ProgramState.DONE,
                    metrics={"fitness": 0.5 + i * 0.01, "is_valid": 1.0},
                )
                await storage.add(p)
                programs.append(p)

        with BenchmarkTimer() as t_get:
            for p in programs:
                await storage.get(p.id)

        add_avg = t_add.elapsed_ms / k
        get_avg = t_get.elapsed_ms / k
        backend = "redis" if redis_url else "fakeredis"
        print(
            f"BENCHMARK: storage_roundtrip code={code_size}B ({backend}): "
            f"add={add_avg:.3f}ms get={get_avg:.3f}ms"
        )
        await cleanup_storage(storage)


class TestComplexityScaling:
    """AST complexity analysis cost vs code size."""

    def test_complexity_vs_code_size(self, code_size: int) -> None:
        code = make_realistic_code(code_size)

        k = 200
        with BenchmarkTimer() as t:
            for _ in range(k):
                compute_numerical_complexity(code)

        avg_ms = t.elapsed_ms / k
        print(
            f"BENCHMARK: complexity_analysis code={code_size}B: "
            f"{avg_ms:.3f}ms/call ({t.elapsed_ms:.1f}ms for {k} calls)"
        )


class TestBatchMgetScaling:
    """mget() cost with large code — simulates get_all_by_status."""

    async def test_batch_mget(self, redis_url: str | None) -> None:
        server = None if redis_url else fakeredis.FakeServer()
        storage = make_storage(server=server, redis_url=redis_url)
        code = make_realistic_code(2000)  # ~2KB realistic code

        ids = []
        for i in range(100):
            p = Program(
                code=code,
                state=ProgramState.DONE,
                metrics={"fitness": float(i), "is_valid": 1.0},
            )
            await storage.add(p)
            ids.append(p.id)

        k = 20
        with BenchmarkTimer() as t:
            for _ in range(k):
                await storage.mget(ids)

        avg_ms = t.elapsed_ms / k
        backend = "redis" if redis_url else "fakeredis"
        print(f"BENCHMARK: mget 100 programs code=2KB ({backend}): {avg_ms:.2f}ms/call")
        await cleanup_storage(storage)
