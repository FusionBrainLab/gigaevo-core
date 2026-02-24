"""Tests for Python executor subprocess pool: WorkerPool lifecycle, run_exec_runner,
timeout handling, error propagation, and one-shot fallback."""

from __future__ import annotations

import asyncio

import pytest

from gigaevo.programs.stages.python_executors.wrapper import (
    ExecRunnerError,
    WorkerPool,
    run_exec_runner,
)

# ---------------------------------------------------------------------------
# run_exec_runner — basic execution
# ---------------------------------------------------------------------------


class TestRunExecRunner:
    async def test_simple_function_returns_result(self) -> None:
        """Execute a simple function and get the return value."""
        code = "def run_code(): return 42"
        result, stdout, stderr = await run_exec_runner(
            code=code, function_name="run_code", timeout=10
        )
        assert result == 42

    async def test_function_with_args(self) -> None:
        code = "def add(a, b): return a + b"
        result, _, _ = await run_exec_runner(
            code=code,
            function_name="add",
            args=[3, 7],
            timeout=10,
        )
        assert result == 10

    async def test_function_with_kwargs(self) -> None:
        code = "def greet(name='world'): return f'hello {name}'"
        result, _, _ = await run_exec_runner(
            code=code,
            function_name="greet",
            kwargs={"name": "test"},
            timeout=10,
        )
        assert result == "hello test"

    async def test_returns_complex_object(self) -> None:
        """Complex return values (dict, list, nested) survive serialization."""
        code = "def run_code(): return {'a': [1, 2], 'b': {'nested': True}}"
        result, _, _ = await run_exec_runner(
            code=code, function_name="run_code", timeout=10
        )
        assert result == {"a": [1, 2], "b": {"nested": True}}

    async def test_returns_numpy_array(self) -> None:
        code = """
import numpy as np
def run_code():
    return np.array([1.0, 2.0, 3.0])
"""
        result, _, _ = await run_exec_runner(
            code=code, function_name="run_code", timeout=10
        )
        import numpy as np

        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# run_exec_runner — error handling
# ---------------------------------------------------------------------------


class TestRunExecRunnerErrors:
    async def test_syntax_error_raises_exec_runner_error(self) -> None:
        code = "def run_code(\n  return 42"  # syntax error
        with pytest.raises(ExecRunnerError) as exc_info:
            await run_exec_runner(code=code, function_name="run_code", timeout=10)
        assert "SyntaxError" in exc_info.value.stderr

    async def test_runtime_error_raises_exec_runner_error(self) -> None:
        code = "def run_code(): raise ValueError('test error')"
        with pytest.raises(ExecRunnerError) as exc_info:
            await run_exec_runner(code=code, function_name="run_code", timeout=10)
        assert "ValueError" in exc_info.value.stderr
        assert "test error" in exc_info.value.stderr

    async def test_missing_function_raises(self) -> None:
        code = "def other_func(): return 1"
        with pytest.raises(ExecRunnerError) as exc_info:
            await run_exec_runner(code=code, function_name="nonexistent", timeout=10)
        assert (
            "not found" in exc_info.value.stderr
            or "not callable" in exc_info.value.stderr
        )

    async def test_timeout_raises(self) -> None:
        code = """
import time
def run_code():
    time.sleep(30)
    return 0
"""
        with pytest.raises((asyncio.TimeoutError, ExecRunnerError)):
            await run_exec_runner(code=code, function_name="run_code", timeout=1)

    async def test_output_too_large_raises(self) -> None:
        code = "def run_code(): return 'x' * 1000000"
        with pytest.raises(ExecRunnerError) as exc_info:
            await run_exec_runner(
                code=code,
                function_name="run_code",
                timeout=10,
                max_output_size=1024,  # 1KB limit
            )
        assert "OutputTooLarge" in exc_info.value.stderr


# ---------------------------------------------------------------------------
# WorkerPool
# ---------------------------------------------------------------------------


class TestWorkerPool:
    def test_default_max_workers(self) -> None:
        import os

        pool = WorkerPool()
        cpu = os.cpu_count() or 4
        expected = max(1, min(32, cpu * 2))
        assert pool.max_workers == expected

    def test_custom_max_workers(self) -> None:
        pool = WorkerPool(max_workers=4)
        assert pool.max_workers == 4

    async def test_worker_reuse(self) -> None:
        """A returned worker can be reused for the next request."""
        pool = WorkerPool(max_workers=1)
        code = "def run_code(): return 1"

        result1, _, _ = await run_exec_runner(
            code=code, function_name="run_code", timeout=10, pool=pool
        )
        result2, _, _ = await run_exec_runner(
            code=code, function_name="run_code", timeout=10, pool=pool
        )

        assert result1 == 1
        assert result2 == 1

    async def test_parallel_execution_with_pool(self) -> None:
        """Multiple tasks run concurrently with a pool."""
        pool = WorkerPool(max_workers=4)
        code = """
import time
def run_code(n):
    time.sleep(0.1)
    return n * 2
"""
        tasks = [
            run_exec_runner(
                code=code,
                function_name="run_code",
                args=[i],
                timeout=10,
                pool=pool,
            )
            for i in range(4)
        ]
        results = await asyncio.gather(*tasks)
        values = sorted([r[0] for r in results])
        assert values == [0, 2, 4, 6]


# ---------------------------------------------------------------------------
# Worker error recovery — one-shot fallback
# ---------------------------------------------------------------------------


class TestWorkerRecovery:
    async def test_error_in_worker_doesnt_break_pool(self) -> None:
        """After a worker error, the pool can still serve requests."""
        pool = WorkerPool(max_workers=2)

        # First request: errors
        bad_code = "def run_code(): raise SystemExit(1)"
        with pytest.raises(ExecRunnerError):
            await run_exec_runner(
                code=bad_code, function_name="run_code", timeout=10, pool=pool
            )

        # Second request: succeeds (pool creates new worker or falls back)
        good_code = "def run_code(): return 'ok'"
        result, _, _ = await run_exec_runner(
            code=good_code, function_name="run_code", timeout=10, pool=pool
        )
        assert result == "ok"

    async def test_exec_runner_error_attributes(self) -> None:
        code = "def run_code(): raise RuntimeError('boom')"
        with pytest.raises(ExecRunnerError) as exc_info:
            await run_exec_runner(code=code, function_name="run_code", timeout=10)
        err = exc_info.value
        assert err.returncode == 1
        assert "RuntimeError" in err.stderr
        assert "boom" in err.stderr


# ---------------------------------------------------------------------------
# PythonCodeExecutor stage class
# ---------------------------------------------------------------------------


class TestPythonCodeExecutorStage:
    async def test_compute_success(self) -> None:
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import (
            CallProgramFunction,
        )

        stage = CallProgramFunction(function_name="solve", timeout=10)
        stage.attach_inputs({})
        prog = Program(code="def solve(): return 42")

        result = await stage.compute(prog)
        assert result.data == 42

    async def test_compute_failure_returns_stage_result(self) -> None:
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import (
            CallProgramFunction,
        )

        stage = CallProgramFunction(function_name="solve", timeout=10)
        stage.attach_inputs({})
        prog = Program(code="def solve(): raise ValueError('nope')")

        result = await stage.compute(prog)
        # Should return a ProgramStageResult failure, not raise
        from gigaevo.programs.core_types import ProgramStageResult

        assert isinstance(result, ProgramStageResult)
        assert result.status.value == "failed"
        assert "ValueError" in result.error.traceback
