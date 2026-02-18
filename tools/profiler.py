#!/usr/bin/env python3
"""
Profiler script to measure throughput of key GigaEvo operations.

Usage:
    python tools/profiler.py --redis-url redis://localhost:6379/15

This script benchmarks:
- Redis read/write operations (single, batch)
- Status set operations (add, count, get_all_by_status)
- Program serialization/deserialization
- Python executor stage (minimal code e.g. 1+1) for debugging executor implementations
- Concurrent operations (parallel reads/writes)
- DAG construction and execution
- Stage execution overhead
- Realistic throughput simulation (mixed workload)
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import statistics
import time
from typing import Any, Callable, Coroutine
import uuid

from loguru import logger

from gigaevo.database.redis.config import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.programs.program import Program
from gigaevo.programs.stages.python_executors.execution import (
    CallProgramFunctionWithFixedArgs,
)

# Configure minimal logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), format="{message}", level="INFO")


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    total_time_s: float
    times_ms: list[float] = field(default_factory=list)

    @property
    def ops_per_sec(self) -> float:
        return self.iterations / self.total_time_s if self.total_time_s > 0 else 0

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0

    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.times_ms:
            return 0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def std_ms(self) -> float:
        if not self.times_ms or len(self.times_ms) < 2:
            return 0
        return statistics.stdev(self.times_ms)

    def __str__(self) -> str:
        return (
            f"{self.name:40} | "
            f"{self.ops_per_sec:>8.1f} ops/s | "
            f"avg={self.avg_ms:>6.2f}ms | "
            f"std={self.std_ms:>6.2f}ms | "
            f"p50={self.p50_ms:>6.2f}ms | "
            f"p95={self.p95_ms:>6.2f}ms | "
            f"p99={self.p99_ms:>6.2f}ms"
        )


async def benchmark(
    name: str,
    fn: Callable[[], Coroutine[Any, Any, Any]],
    iterations: int = 100,
    warmup: int = 5,
) -> BenchmarkResult:
    """Run a benchmark."""
    # Warmup
    for _ in range(warmup):
        await fn()

    times_ms: list[float] = []
    start = time.perf_counter()

    for _ in range(iterations):
        t0 = time.perf_counter()
        await fn()
        times_ms.append((time.perf_counter() - t0) * 1000)

    total = time.perf_counter() - start

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_s=total,
        times_ms=times_ms,
    )


async def run_redis_benchmarks(
    redis_url: str, iterations: int = 100
) -> list[BenchmarkResult]:
    """Run Redis-specific benchmarks."""

    results: list[BenchmarkResult] = []

    # Setup
    config = RedisProgramStorageConfig(
        redis_url=redis_url,
        key_prefix=f"profiler_{uuid.uuid4().hex[:8]}",
        read_only=False,
    )

    storage = RedisProgramStorage(config)
    await storage._conn.get()  # Establish connection

    try:
        # 1. Single program add
        async def add_single():
            p = Program(code="def run_code(): return 42")
            await storage.add(p)
            return p.id

        result = await benchmark("Redis: add single program", add_single, iterations)
        results.append(result)

        # 2. Single program get
        test_program = Program(code="def run_code(): return 42")
        await storage.add(test_program)

        async def get_single():
            return await storage.get(test_program.id)

        result = await benchmark("Redis: get single program", get_single, iterations)
        results.append(result)

        # 3. Batch mget (10 programs)
        batch_ids = []
        for _ in range(10):
            p = Program(code="def run_code(): return 42")
            await storage.add(p)
            batch_ids.append(p.id)

        async def mget_batch():
            return await storage.mget(batch_ids)

        result = await benchmark("Redis: mget 10 programs", mget_batch, iterations)
        results.append(result)

        # 4. Count by status (uses SCARD)
        async def count_status():
            return await storage.count_by_status("fresh")

        result = await benchmark(
            "Redis: count_by_status (SCARD)", count_status, iterations
        )
        results.append(result)

        # 5. Get all by status
        async def get_all_status():
            return await storage.get_all_by_status("fresh")

        result = await benchmark("Redis: get_all_by_status", get_all_status, iterations)
        results.append(result)

        # 6. Size (SCAN)
        async def size_scan():
            return await storage.size()

        result = await benchmark("Redis: size (SCAN)", size_scan, iterations)
        results.append(result)

        # 7. Exists check
        async def exists_check():
            return await storage.exists(test_program.id)

        result = await benchmark("Redis: exists check", exists_check, iterations)
        results.append(result)

        # 8. Update program
        async def update_program():
            test_program.metrics["test"] = time.time()
            await storage.update(test_program)

        result = await benchmark("Redis: update program", update_program, iterations)
        results.append(result)

        # 9. Update with stage_results (simulates DAG stage completion)
        from gigaevo.programs.core_types import ProgramStageResult, StageState

        stage_counter = [0]

        async def update_with_stage_result():
            stage_counter[0] += 1
            stage_name = f"stage_{stage_counter[0]}"
            test_program.stage_results[stage_name] = ProgramStageResult(
                status=StageState.COMPLETED
            )
            await storage.update(test_program)

        result = await benchmark(
            "Redis: update with stage_result", update_with_stage_result, iterations
        )
        results.append(result)

        # 10. Concurrent updates (simulates parallel stages)
        async def concurrent_stage_updates():
            async def update_stage(stage_num: int):
                # Each "stage" updates a different key
                test_program.stage_results[f"parallel_{stage_num}"] = (
                    ProgramStageResult(status=StageState.COMPLETED)
                )
                await storage.update(test_program)

            await asyncio.gather(*[update_stage(i) for i in range(3)])

        result = await benchmark(
            "Redis: 3 concurrent stage updates",
            concurrent_stage_updates,
            iterations // 2,
        )
        results.append(result)

    finally:
        # Cleanup
        async def cleanup(r):
            keys = [k async for k in r.scan_iter(f"{config.key_prefix}:*")]
            if keys:
                await r.delete(*keys)

        await storage._conn.execute("cleanup", cleanup)
        await storage.close()

    return results


async def run_serialization_benchmarks(iterations: int = 1000) -> list[BenchmarkResult]:
    """Benchmark program serialization/deserialization."""
    from gigaevo.programs.core_types import ProgramStageResult, StageState
    from gigaevo.programs.program import Program

    results: list[BenchmarkResult] = []

    # Create a realistic program with stage results
    program = Program(
        code="def run_code():\n    return [i**2 for i in range(100)]",
        metrics={"fitness": 0.95, "is_valid": 1.0, "complexity": 42.5},
        metadata={"mutation_context": "test context " * 100},
    )
    program.stage_results["execution"] = ProgramStageResult(status=StageState.COMPLETED)
    program.stage_results["validation"] = ProgramStageResult(
        status=StageState.COMPLETED
    )

    # 1. Serialize to dict
    async def serialize():
        return program.to_dict()

    result = await benchmark("Serialization: Program.to_dict()", serialize, iterations)
    results.append(result)

    # 2. Deserialize from dict
    data = program.to_dict()

    async def deserialize():
        return Program.from_dict(data)

    result = await benchmark(
        "Deserialization: Program.from_dict()", deserialize, iterations
    )
    results.append(result)

    # 3. JSON round-trip
    import json

    async def json_roundtrip():
        d = program.to_dict()
        s = json.dumps(d)
        return Program.from_dict(json.loads(s))

    result = await benchmark("JSON round-trip", json_roundtrip, iterations)
    results.append(result)

    return results


async def run_python_executor_benchmarks(iterations: int = 50) -> list[BenchmarkResult]:
    """Benchmark Python executor stage with minimal code (e.g. 1+1) for debugging executor implementations."""

    results: list[BenchmarkResult] = []

    # Minimal program: single expression, no I/O
    minimal_code = "def run_code():\n    return 1 + 1"
    program = Program(code=minimal_code)

    # Stage with VoidInput (no context), default timeout (subprocess spawn + run)
    stage = CallProgramFunctionWithFixedArgs(timeout=30)
    stage.attach_inputs({})

    async def run_single_execution():
        return await stage.execute(program)

    result = await benchmark(
        "Python executor: run_code() -> 1+1 (subprocess)",
        run_single_execution,
        iterations=iterations,
        warmup=3,
    )
    results.append(result)

    return results


async def run_concurrent_benchmarks(
    redis_url: str, concurrency: int = 10
) -> list[BenchmarkResult]:
    """Benchmark concurrent operations."""

    results: list[BenchmarkResult] = []

    config = RedisProgramStorageConfig(
        redis_url=redis_url,
        key_prefix=f"profiler_concurrent_{uuid.uuid4().hex[:8]}",
        read_only=False,
    )

    storage = RedisProgramStorage(config)
    await storage._conn.get()

    try:
        # Pre-populate
        program_ids = []
        for _ in range(100):
            p = Program(code="def run_code(): return 42")
            await storage.add(p)
            program_ids.append(p.id)

        # Concurrent reads
        async def concurrent_reads():
            tasks = [storage.get(pid) for pid in program_ids[:concurrency]]
            await asyncio.gather(*tasks)

        result = await benchmark(
            f"Concurrent: {concurrency} parallel gets",
            concurrent_reads,
            iterations=50,
        )
        results.append(result)

        # Concurrent writes
        async def concurrent_writes():
            programs = [
                Program(code="def run_code(): return 42") for _ in range(concurrency)
            ]
            tasks = [storage.add(p) for p in programs]
            await asyncio.gather(*tasks)

        result = await benchmark(
            f"Concurrent: {concurrency} parallel adds",
            concurrent_writes,
            iterations=50,
        )
        results.append(result)

    finally:

        async def cleanup(r):
            keys = [k async for k in r.scan_iter(f"{config.key_prefix}:*")]
            if keys:
                await r.delete(*keys)

        await storage._conn.execute("cleanup", cleanup)
        await storage.close()

    return results


async def run_dag_benchmarks(
    redis_url: str, iterations: int = 20
) -> list[BenchmarkResult]:
    """Benchmark DAG construction and execution."""
    from gigaevo.database.redis.config import RedisProgramStorageConfig
    from gigaevo.database.redis_program_storage import RedisProgramStorage
    from gigaevo.database.state_manager import ProgramStateManager
    from gigaevo.programs.core_types import StageIO, VoidInput
    from gigaevo.programs.dag.automata import DataFlowEdge
    from gigaevo.programs.dag.dag import DAG
    from gigaevo.programs.program import Program
    from gigaevo.programs.stages.base import Stage

    results: list[BenchmarkResult] = []

    # Create mock stages for benchmarking
    class MockOutput(StageIO):
        value: int = 42

    class FastStage(Stage):
        """A stage that completes instantly."""

        InputsModel = VoidInput
        OutputModel = MockOutput

        async def compute(self, program: Program) -> MockOutput:
            return MockOutput(value=42)

    class ChainedInput(StageIO):
        data: MockOutput

    class ChainedStage(Stage):
        """A stage that takes input from another stage."""

        InputsModel = ChainedInput
        OutputModel = MockOutput

        async def compute(self, program: Program) -> MockOutput:
            return MockOutput(value=self.params.data.value + 1)

    # Setup
    config = RedisProgramStorageConfig(
        redis_url=redis_url,
        key_prefix=f"profiler_dag_{uuid.uuid4().hex[:8]}",
        read_only=False,
    )

    storage = RedisProgramStorage(config)
    await storage._conn.get()
    state_manager = ProgramStateManager(storage)

    # Create a null writer for benchmarks
    class NullWriter:
        def scalar(self, *args, **kwargs):
            pass

        def hist(self, *args, **kwargs):
            pass

        def text(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def bind(self, **kwargs):
            return self

    writer = NullWriter()

    try:
        # 1. DAG construction (simple - 3 stages)
        async def build_simple_dag():
            nodes = {
                "stage1": FastStage(timeout=10.0),
                "stage2": FastStage(timeout=10.0),
                "stage3": FastStage(timeout=10.0),
            }
            edges = []
            return DAG(
                nodes=nodes,
                data_flow_edges=edges,
                execution_order_deps=None,
                state_manager=state_manager,
                writer=writer,
            )

        result = await benchmark(
            "DAG: construct 3 independent stages", build_simple_dag, iterations * 5
        )
        results.append(result)

        # 2. DAG construction (chained - 5 stages)
        async def build_chained_dag():
            nodes = {
                "start": FastStage(timeout=10.0),
                "chain1": ChainedStage(timeout=10.0),
                "chain2": ChainedStage(timeout=10.0),
                "chain3": ChainedStage(timeout=10.0),
                "chain4": ChainedStage(timeout=10.0),
            }
            edges = [
                DataFlowEdge.create("start", "chain1", "data"),
                DataFlowEdge.create("chain1", "chain2", "data"),
                DataFlowEdge.create("chain2", "chain3", "data"),
                DataFlowEdge.create("chain3", "chain4", "data"),
            ]
            return DAG(
                nodes=nodes,
                data_flow_edges=edges,
                execution_order_deps=None,
                state_manager=state_manager,
                writer=writer,
            )

        result = await benchmark(
            "DAG: construct 5 chained stages", build_chained_dag, iterations * 5
        )
        results.append(result)

        # 3. Full DAG run (3 independent fast stages)
        async def run_simple_dag():
            dag = await build_simple_dag()
            program = Program(code="def run_code(): return 42")
            await storage.add(program)
            try:
                await dag.run(program)
            finally:
                # Cleanup DAG resources
                dag.automata.topology.nodes.clear()
                dag.automata.topology = None
                dag.automata = None

        result = await benchmark(
            "DAG: run 3 independent stages", run_simple_dag, iterations
        )
        results.append(result)

        # 4. Full DAG run (5 chained stages)
        async def run_chained_dag():
            dag = await build_chained_dag()
            program = Program(code="def run_code(): return 42")
            await storage.add(program)
            try:
                await dag.run(program)
            finally:
                dag.automata.topology.nodes.clear()
                dag.automata.topology = None
                dag.automata = None

        result = await benchmark(
            "DAG: run 5 chained stages", run_chained_dag, iterations
        )
        results.append(result)

        # 5. Stage execution isolation (single stage, no DAG overhead)
        async def run_single_stage():
            stage = FastStage(timeout=10.0)
            stage.attach_inputs({})
            program = Program(code="def run_code(): return 42")
            return await stage.execute(program)

        result = await benchmark(
            "Stage: single FastStage execute", run_single_stage, iterations * 10
        )
        results.append(result)

        # 6. Program state updates during DAG
        test_program = Program(code="def run_code(): return 42")
        await storage.add(test_program)

        async def update_program_state():
            await state_manager.update_program(test_program)

        result = await benchmark(
            "DAG: state_manager.update_program", update_program_state, iterations * 2
        )
        results.append(result)

        # 7. mark_stage_running (in-memory only, no Redis write)
        async def mark_running():
            await state_manager.mark_stage_running(test_program, "test_stage")

        result = await benchmark(
            "mark_stage_running (in-memory)", mark_running, iterations * 5
        )
        results.append(result)

    finally:

        async def cleanup(r):
            keys = [k async for k in r.scan_iter(f"{config.key_prefix}:*")]
            if keys:
                await r.delete(*keys)

        await storage._conn.execute("cleanup", cleanup)
        await storage.close()

    return results


async def run_throughput_simulation(
    redis_url: str, duration_seconds: float = 5.0
) -> dict[str, Any]:
    """Simulate realistic throughput over time."""
    from gigaevo.database.redis.config import RedisProgramStorageConfig
    from gigaevo.database.redis_program_storage import RedisProgramStorage
    from gigaevo.programs.program import Program

    config = RedisProgramStorageConfig(
        redis_url=redis_url,
        key_prefix=f"profiler_throughput_{uuid.uuid4().hex[:8]}",
        read_only=False,
    )

    storage = RedisProgramStorage(config)
    await storage._conn.get()

    results = {
        "duration_s": duration_seconds,
        "programs_added": 0,
        "programs_read": 0,
        "errors": 0,
    }

    program_ids: list[str] = []
    start = time.perf_counter()

    try:
        # Simulate mixed workload
        while (time.perf_counter() - start) < duration_seconds:
            # Add a program
            p = Program(code="def run_code(): return 42")
            await storage.add(p)
            program_ids.append(p.id)
            results["programs_added"] += 1

            # Read some programs (simulate DAG lookups)
            if len(program_ids) > 5:
                sample_ids = program_ids[-5:]
                await storage.mget(sample_ids)
                results["programs_read"] += 5

            # Check counts (simulates _has_active_dags)
            await storage.count_by_status("fresh")

    except Exception as e:
        results["errors"] += 1
        results["error_msg"] = str(e)

    finally:
        elapsed = time.perf_counter() - start
        results["actual_duration_s"] = elapsed
        results["add_throughput"] = results["programs_added"] / elapsed
        results["read_throughput"] = results["programs_read"] / elapsed

        async def cleanup(r):
            keys = [k async for k in r.scan_iter(f"{config.key_prefix}:*")]
            if keys:
                await r.delete(*keys)

        await storage._conn.execute("cleanup", cleanup)
        await storage.close()

    return results


def print_results(title: str, results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a table."""
    print(f"\n{'=' * 110}")
    print(f" {title}")
    print(f"{'=' * 110}")
    print(
        f"{'Benchmark':40} | {'Throughput':>12} | "
        f"{'Avg':>9} | {'Std':>9} | {'P50':>9} | {'P95':>9} | {'P99':>9}"
    )
    print("-" * 110)
    for r in results:
        print(r)
    print()


async def main(redis_url: str, skip_redis: bool = False) -> None:
    """Run all benchmarks."""
    print("\n" + "=" * 100)
    print(" GigaEvo Performance Profiler")
    print("=" * 100)

    # Serialization benchmarks (no Redis needed)
    print("\nRunning serialization benchmarks...")
    ser_results = await run_serialization_benchmarks(iterations=1000)
    print_results("Serialization Benchmarks", ser_results)

    # Python executor benchmarks (no Redis needed) – for debugging executor implementations
    print("\nRunning Python executor benchmarks (minimal code 1+1)...")
    executor_results = await run_python_executor_benchmarks(iterations=50)
    print_results("Python Executor Benchmarks", executor_results)

    if not skip_redis:
        # Redis benchmarks
        print(f"\nRunning Redis benchmarks (url={redis_url})...")
        try:
            redis_results = await run_redis_benchmarks(redis_url, iterations=100)
            print_results("Redis Operation Benchmarks", redis_results)

            # Concurrent benchmarks
            print("\nRunning concurrency benchmarks...")
            concurrent_results = await run_concurrent_benchmarks(
                redis_url, concurrency=10
            )
            print_results("Concurrency Benchmarks", concurrent_results)

            # DAG benchmarks
            print("\nRunning DAG benchmarks...")
            dag_results = await run_dag_benchmarks(redis_url, iterations=20)
            print_results("DAG Benchmarks", dag_results)

            # Throughput simulation
            print("\nRunning throughput simulation (5 seconds)...")
            throughput = await run_throughput_simulation(
                redis_url, duration_seconds=5.0
            )
            print("\n" + "=" * 100)
            print(" Throughput Simulation Results")
            print("=" * 100)
            print(f"  Duration: {throughput['actual_duration_s']:.2f}s")
            print(f"  Programs added: {throughput['programs_added']}")
            print(f"  Programs read: {throughput['programs_read']}")
            print(f"  Add throughput: {throughput['add_throughput']:.1f} programs/s")
            print(f"  Read throughput: {throughput['read_throughput']:.1f} reads/s")
            if throughput.get("errors"):
                print(f"  Errors: {throughput['errors']}")
            print()

        except Exception as e:
            import traceback

            print(f"\n[ERROR] Benchmarks failed: {e}")
            traceback.print_exc()
            print("  Is Redis running?")

    print("\n" + "=" * 100)
    print(" Summary")
    print("=" * 100)
    print("Key metrics to watch:")
    print("  - Redis single get/add: should be < 1ms avg")
    print("  - count_by_status (SCARD): should be much faster than get_all_by_status")
    print("  - Concurrent operations: check for lock contention")
    print("  - DAG run time: dominated by stage execution + Redis state updates")
    print("  - Throughput simulation: realistic mixed workload performance")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GigaEvo Performance Profiler")
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/15",
        help="Redis URL for benchmarks (default: redis://localhost:6379/15)",
    )
    parser.add_argument(
        "--skip-redis",
        action="store_true",
        help="Skip Redis benchmarks (only run serialization)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.redis_url, args.skip_redis))
