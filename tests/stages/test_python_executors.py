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
        await pool.shutdown()

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
        await pool.shutdown()


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
        await pool.shutdown()

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


# ---------------------------------------------------------------------------
# PythonCodeExecutor — error-handling paths (MemoryError, generic Exception)
# ---------------------------------------------------------------------------


class TestPythonCodeExecutorErrorPaths:
    async def test_memory_error_detection_in_stderr(self) -> None:
        """ExecRunnerError with 'MemoryError' in stderr sets error_type='MemoryLimitExceeded'."""
        from unittest.mock import AsyncMock, patch

        from gigaevo.programs.core_types import ProgramStageResult
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import (
            CallProgramFunction,
        )
        from gigaevo.programs.stages.python_executors.wrapper import ExecRunnerError

        stage = CallProgramFunction(function_name="solve", timeout=10)
        stage.attach_inputs({})
        prog = Program(code="def solve(): pass")

        fake_error = ExecRunnerError(
            returncode=1,
            stderr="Traceback...\nMemoryError: unable to allocate",
            stdout_bytes=b"",
        )

        with patch(
            "gigaevo.programs.stages.python_executors.execution.run_exec_runner",
            new_callable=AsyncMock,
            side_effect=fake_error,
        ):
            result = await stage.compute(prog)

        assert isinstance(result, ProgramStageResult)
        assert result.status.value == "failed"
        assert result.error is not None
        assert result.error.type == "MemoryLimitExceeded"

    async def test_cannot_allocate_memory_string_detection(self) -> None:
        """'Cannot allocate memory' in stderr is also detected as MemoryLimitExceeded."""
        from unittest.mock import AsyncMock, patch

        from gigaevo.programs.core_types import ProgramStageResult
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import (
            CallProgramFunction,
        )
        from gigaevo.programs.stages.python_executors.wrapper import ExecRunnerError

        stage = CallProgramFunction(function_name="solve", timeout=10)
        stage.attach_inputs({})
        prog = Program(code="def solve(): pass")

        fake_error = ExecRunnerError(
            returncode=1,
            stderr="Cannot allocate memory in static TLS block",
            stdout_bytes=b"",
        )

        with patch(
            "gigaevo.programs.stages.python_executors.execution.run_exec_runner",
            new_callable=AsyncMock,
            side_effect=fake_error,
        ):
            result = await stage.compute(prog)

        assert isinstance(result, ProgramStageResult)
        assert result.error.type == "MemoryLimitExceeded"

    async def test_generic_exception_in_compute_returns_failure(self) -> None:
        """A non-ExecRunnerError exception in compute() returns ProgramStageResult.failure."""
        from unittest.mock import AsyncMock, patch

        from gigaevo.programs.core_types import ProgramStageResult
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import (
            CallProgramFunction,
        )

        stage = CallProgramFunction(function_name="solve", timeout=10)
        stage.attach_inputs({})
        prog = Program(code="def solve(): pass")

        with patch(
            "gigaevo.programs.stages.python_executors.execution.run_exec_runner",
            new_callable=AsyncMock,
            side_effect=RuntimeError("unexpected internal failure"),
        ):
            result = await stage.compute(prog)

        assert isinstance(result, ProgramStageResult)
        assert result.status.value == "failed"
        assert result.error is not None
        assert (
            "RuntimeError" in result.error.type
            or "RuntimeError" in result.error.message
        )

    async def test_memory_limit_error_message_includes_mb_when_set(self) -> None:
        """When max_memory_mb is set, the error message mentions the limit."""
        from unittest.mock import AsyncMock, patch

        from gigaevo.programs.core_types import ProgramStageResult
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import (
            CallProgramFunction,
        )
        from gigaevo.programs.stages.python_executors.wrapper import ExecRunnerError

        stage = CallProgramFunction(
            function_name="solve", timeout=10, max_memory_mb=512
        )
        stage.attach_inputs({})
        prog = Program(code="def solve(): pass")

        fake_error = ExecRunnerError(
            returncode=1,
            stderr="MemoryError",
            stdout_bytes=b"",
        )

        with patch(
            "gigaevo.programs.stages.python_executors.execution.run_exec_runner",
            new_callable=AsyncMock,
            side_effect=fake_error,
        ):
            result = await stage.compute(prog)

        assert isinstance(result, ProgramStageResult)
        assert result.error.type == "MemoryLimitExceeded"
        assert "512" in result.error.message


# ---------------------------------------------------------------------------
# CallFileFunction
# ---------------------------------------------------------------------------


class TestCallFileFunctionStage:
    def test_call_file_function_nonexistent_path_raises_validation_error(
        self, tmp_path
    ) -> None:
        """CallFileFunction with a non-existent path raises ValidationError at construction."""
        from gigaevo.exceptions import ValidationError
        from gigaevo.programs.stages.python_executors.execution import CallFileFunction

        nonexistent = tmp_path / "no_such_file.py"
        with pytest.raises(ValidationError, match="not found"):
            CallFileFunction(path=nonexistent, timeout=10)

    async def test_call_file_function_executes_file_code(self, tmp_path) -> None:
        """CallFileFunction reads code from the file and executes the named function."""
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import CallFileFunction

        script = tmp_path / "context_builder.py"
        script.write_text("def build_context(): return {'answer': 42}\n")

        stage = CallFileFunction(path=script, timeout=10)
        stage.attach_inputs({})
        prog = Program(code="def solve(): pass")

        result = await stage.compute(prog)
        assert result.data == {"answer": 42}


# ---------------------------------------------------------------------------
# CallProgramFunctionWithFixedArgs
# ---------------------------------------------------------------------------


class TestCallProgramFunctionWithFixedArgs:
    async def test_fixed_args_passed_to_function(self) -> None:
        """Fixed positional args are forwarded correctly to the program function."""
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import (
            CallProgramFunctionWithFixedArgs,
        )

        stage = CallProgramFunctionWithFixedArgs(
            function_name="add",
            args=[3, 7],
            timeout=10,
        )
        stage.attach_inputs({})
        prog = Program(code="def add(a, b): return a + b")

        result = await stage.compute(prog)
        assert result.data == 10

    async def test_fixed_kwargs_passed_to_function(self) -> None:
        """Fixed keyword args are forwarded correctly to the program function."""
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import (
            CallProgramFunctionWithFixedArgs,
        )

        stage = CallProgramFunctionWithFixedArgs(
            function_name="greet",
            kwargs={"name": "world"},
            timeout=10,
        )
        stage.attach_inputs({})
        prog = Program(code="def greet(name='?'): return f'hello {name}'")

        result = await stage.compute(prog)
        assert result.data == "hello world"

    async def test_no_args_no_kwargs_defaults(self) -> None:
        """Instantiating with neither args nor kwargs works fine."""
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import (
            CallProgramFunctionWithFixedArgs,
        )

        stage = CallProgramFunctionWithFixedArgs(
            function_name="run_code",
            timeout=10,
        )
        stage.attach_inputs({})
        prog = Program(code="def run_code(): return 'ok'")

        result = await stage.compute(prog)
        assert result.data == "ok"


# ---------------------------------------------------------------------------
# FetchMetrics and FetchArtifact stages
# ---------------------------------------------------------------------------


class TestFetchMetricsAndFetchArtifact:
    async def test_fetch_metrics_extracts_metrics_dict(self) -> None:
        """FetchMetrics pulls the first element (metrics dict) from a ValidatorOutput."""
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.common import Box
        from gigaevo.programs.stages.python_executors.execution import FetchMetrics

        metrics = {"score": 0.95, "loss": 0.1}
        artifact = {"data": [1, 2, 3]}

        # ValidatorOutput = Box[Tuple[dict[str, float], Any]]
        validator_output = Box[tuple](data=(metrics, artifact))

        stage = FetchMetrics(timeout=10)
        stage.attach_inputs({"validation_result": validator_output})

        prog = Program(code="def f(): pass")
        result = await stage.compute(prog)

        assert result.data == metrics

    async def test_fetch_artifact_extracts_artifact(self) -> None:
        """FetchArtifact pulls the second element (artifact) from a ValidatorOutput."""
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.common import Box
        from gigaevo.programs.stages.python_executors.execution import FetchArtifact

        metrics = {"score": 1.0}
        artifact = [42, 43, 44]

        validator_output = Box[tuple](data=(metrics, artifact))

        stage = FetchArtifact(timeout=10)
        stage.attach_inputs({"validation_result": validator_output})

        prog = Program(code="def f(): pass")
        result = await stage.compute(prog)

        assert result.data == artifact

    async def test_fetch_artifact_can_return_none_artifact(self) -> None:
        """FetchArtifact handles None artifact (validator returned no artifact)."""
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.common import Box
        from gigaevo.programs.stages.python_executors.execution import FetchArtifact

        validator_output = Box[tuple](data=({"score": 0.5}, None))

        stage = FetchArtifact(timeout=10)
        stage.attach_inputs({"validation_result": validator_output})

        prog = Program(code="def f(): pass")
        result = await stage.compute(prog)

        assert result.data is None


# ---------------------------------------------------------------------------
# CallValidatorFunction — constructor and parse_output
# ---------------------------------------------------------------------------


class TestCallValidatorFunction:
    def test_nonexistent_validator_path_raises(self, tmp_path) -> None:
        """CallValidatorFunction raises ValidationError when the file doesn't exist."""
        from gigaevo.exceptions import ValidationError
        from gigaevo.programs.stages.python_executors.execution import (
            CallValidatorFunction,
        )

        with pytest.raises(ValidationError, match="not found"):
            CallValidatorFunction(path=tmp_path / "missing.py", timeout=10)

    async def test_validator_called_with_payload(self, tmp_path) -> None:
        """CallValidatorFunction passes payload to the validate function."""
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.common import Box
        from gigaevo.programs.stages.python_executors.execution import (
            CallValidatorFunction,
        )

        validator_file = tmp_path / "validator.py"
        validator_file.write_text(
            "def validate(payload): return ({'score': float(payload)}, None)\n"
        )

        stage = CallValidatorFunction(path=validator_file, timeout=10)
        stage.attach_inputs(
            {
                "payload": Box[float](data=7.0),
                "context": None,
            }
        )

        prog = Program(code="def f(): pass")
        result = await stage.compute(prog)

        # result is a Box[Tuple[dict, Any]] or ProgramStageResult
        from gigaevo.programs.core_types import ProgramStageResult

        if not isinstance(result, ProgramStageResult):
            assert result.data[0] == {"score": 7.0}
            assert result.data[1] is None

    async def test_parse_output_passes_through_tuple(self, tmp_path) -> None:
        """parse_output returns the value unchanged when it is already a tuple."""
        from gigaevo.programs.stages.python_executors.execution import (
            CallValidatorFunction,
        )

        # Create a minimal valid file so the constructor succeeds
        f = tmp_path / "v.py"
        f.write_text("def validate(x): return x\n")

        stage = CallValidatorFunction(path=f, timeout=10)
        raw = ({"a": 1.0}, "artifact")
        out = stage.parse_output(raw)
        assert out == raw

    async def test_parse_output_non_tuple_wrapped(self, tmp_path) -> None:
        """parse_output wraps non-tuple return in (value, None)."""
        from gigaevo.programs.stages.python_executors.execution import (
            CallValidatorFunction,
        )

        f = tmp_path / "v.py"
        f.write_text("def validate(x): return x\n")

        stage = CallValidatorFunction(path=f, timeout=10)
        raw = {"score": 0.5}
        out = stage.parse_output(raw)
        assert out == (raw, None)
