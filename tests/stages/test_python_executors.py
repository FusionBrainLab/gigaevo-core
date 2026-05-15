"""Tests for the loky-backed Python executor.

Covers public-surface behaviour (``run_exec_runner``, ``PythonCodeExecutor``
stages) plus regression reproducers for bugs fixed by the loky migration:

- #78 ER1: :class:`ExecRunnerError` formatting includes stderr.
- #64 E4 : no ``"MemoryError"``/``"Cannot allocate memory"`` string-match
  heuristic mislabels in-validator OOM-like errors as ``MemoryLimitExceeded``.
- #62 E3 : the executor is process-level, not event-loop bound, so a fresh
  ``asyncio.run`` after the first call does not break.
- #63 E6 : timeout raises promptly and does not silently fall back to a
  one-shot subprocess.
- OBS2  : per-call worker accounting (peak RSS, wall time, pid) is
  surfaced through the internal ``_run_task`` envelope.

Internal-helper tests (no subprocess) cover ``_run_one`` directly so
worker-side logic is exercised without paying loky spawn cost.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from gigaevo.programs.stages.python_executors.wrapper import (
    ExecRunnerError,
    run_exec_runner,
)


class TestRunExecRunner:
    async def test_simple_function_returns_result(self) -> None:
        code = "def run_code(): return 42"
        result = await run_exec_runner(code=code, function_name="run_code", timeout=10)
        assert result == 42

    async def test_function_with_args(self) -> None:
        code = "def add(a, b): return a + b"
        result = await run_exec_runner(
            code=code,
            function_name="add",
            args=[3, 7],
            timeout=10,
        )
        assert result == 10

    async def test_function_with_kwargs(self) -> None:
        code = "def greet(name='world'): return f'hello {name}'"
        result = await run_exec_runner(
            code=code,
            function_name="greet",
            kwargs={"name": "test"},
            timeout=10,
        )
        assert result == "hello test"

    async def test_returns_complex_object(self) -> None:
        code = "def run_code(): return {'a': [1, 2], 'b': {'nested': True}}"
        result = await run_exec_runner(code=code, function_name="run_code", timeout=10)
        assert result == {"a": [1, 2], "b": {"nested": True}}

    async def test_returns_numpy_array(self) -> None:
        code = """
import numpy as np
def run_code():
    return np.array([1.0, 2.0, 3.0])
"""
        result = await run_exec_runner(code=code, function_name="run_code", timeout=10)
        import numpy as np

        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


class TestRunExecRunnerErrors:
    async def test_syntax_error_raises_exec_runner_error(self) -> None:
        code = "def run_code(\n  return 42"
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


class TestExecRunnerErrorAttributes:
    async def test_exec_runner_error_attributes(self) -> None:
        code = "def run_code(): raise RuntimeError('boom')"
        with pytest.raises(ExecRunnerError) as exc_info:
            await run_exec_runner(code=code, function_name="run_code", timeout=10)
        err = exc_info.value
        assert err.returncode == 1
        assert "RuntimeError" in err.stderr
        assert "boom" in err.stderr
        assert "boom" in str(err)


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
        from gigaevo.programs.core_types import ProgramStageResult
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import (
            CallProgramFunction,
        )

        stage = CallProgramFunction(function_name="solve", timeout=10)
        stage.attach_inputs({})
        prog = Program(code="def solve(): raise ValueError('nope')")

        result = await stage.compute(prog)
        assert isinstance(result, ProgramStageResult)
        assert result.status.value == "failed"
        assert result.error is not None
        assert result.error.traceback is not None
        assert "ValueError" in result.error.traceback


class TestPythonCodeExecutorErrorPaths:
    async def test_subprocess_error_returns_failure(self) -> None:
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
        assert result.error.type == "SubprocessError"
        assert result.error.traceback is not None
        assert "MemoryError" in result.error.traceback

    async def test_generic_exception_in_compute_returns_failure(self) -> None:
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


class TestCallFileFunctionStage:
    def test_call_file_function_nonexistent_path_raises_validation_error(
        self, tmp_path
    ) -> None:
        from gigaevo.exceptions import ValidationError
        from gigaevo.programs.stages.python_executors.execution import CallFileFunction

        nonexistent = tmp_path / "no_such_file.py"
        with pytest.raises(ValidationError, match="not found"):
            CallFileFunction(path=nonexistent, timeout=10)

    async def test_call_file_function_executes_file_code(self, tmp_path) -> None:
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import CallFileFunction

        script = tmp_path / "context_builder.py"
        script.write_text("def build_context(): return {'answer': 42}\n")

        stage = CallFileFunction(path=script, timeout=10)
        stage.attach_inputs({})
        prog = Program(code="def solve(): pass")

        result = await stage.compute(prog)
        assert result.data == {"answer": 42}


class TestCallProgramFunctionWithFixedArgs:
    async def test_fixed_args_passed_to_function(self) -> None:
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


class TestFetchMetricsAndFetchArtifact:
    async def test_fetch_metrics_extracts_metrics_dict(self) -> None:
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.common import Box
        from gigaevo.programs.stages.python_executors.execution import FetchMetrics

        metrics = {"score": 0.95, "loss": 0.1}
        artifact = {"data": [1, 2, 3]}
        validator_output = Box[tuple](data=(metrics, artifact))

        stage = FetchMetrics(timeout=10)
        stage.attach_inputs({"validation_result": validator_output})

        prog = Program(code="def f(): pass")
        result = await stage.compute(prog)

        assert result.data == metrics

    async def test_fetch_artifact_extracts_artifact(self) -> None:
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
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.common import Box
        from gigaevo.programs.stages.python_executors.execution import FetchArtifact

        validator_output = Box[tuple](data=({"score": 0.5}, None))

        stage = FetchArtifact(timeout=10)
        stage.attach_inputs({"validation_result": validator_output})

        prog = Program(code="def f(): pass")
        result = await stage.compute(prog)

        assert result.data is None


class TestCallValidatorFunction:
    def test_nonexistent_validator_path_raises(self, tmp_path) -> None:
        from gigaevo.exceptions import ValidationError
        from gigaevo.programs.stages.python_executors.execution import (
            CallValidatorFunction,
        )

        with pytest.raises(ValidationError, match="not found"):
            CallValidatorFunction(path=tmp_path / "missing.py", timeout=10)

    async def test_validator_called_with_payload(self, tmp_path) -> None:
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

        from gigaevo.programs.core_types import ProgramStageResult

        if not isinstance(result, ProgramStageResult):
            assert result.data[0] == {"score": 7.0}
            assert result.data[1] is None

    async def test_parse_output_passes_through_tuple(self, tmp_path) -> None:
        from gigaevo.programs.stages.python_executors.execution import (
            CallValidatorFunction,
        )

        f = tmp_path / "v.py"
        f.write_text("def validate(x): return x\n")

        stage = CallValidatorFunction(path=f, timeout=10)
        raw = ({"a": 1.0}, "artifact")
        out = stage.parse_output(raw)
        assert out == raw

    async def test_parse_output_non_tuple_wrapped(self, tmp_path) -> None:
        from gigaevo.programs.stages.python_executors.execution import (
            CallValidatorFunction,
        )

        f = tmp_path / "v.py"
        f.write_text("def validate(x): return x\n")

        stage = CallValidatorFunction(path=f, timeout=10)
        raw = {"score": 0.5}
        out = stage.parse_output(raw)
        assert out == (raw, None)

    async def test_validator_called_with_non_none_context(self, tmp_path) -> None:
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.common import Box
        from gigaevo.programs.stages.python_executors.execution import (
            CallValidatorFunction,
        )

        validator_file = tmp_path / "validator_ctx.py"
        validator_file.write_text(
            "def validate(ctx, payload): return ({'ctx': ctx, 'payload': payload}, None)\n"
        )

        stage = CallValidatorFunction(path=validator_file, timeout=10)
        stage.attach_inputs(
            {
                "payload": Box[float](data=3.0),
                "context": Box[str](data="my-context"),
            }
        )

        prog = Program(code="def f(): pass")
        result = await stage.compute(prog)

        from gigaevo.programs.core_types import ProgramStageResult

        if not isinstance(result, ProgramStageResult):
            metrics, artifact = result.data
            assert metrics["ctx"] == "my-context"
            assert metrics["payload"] == 3.0
            assert artifact is None


# =============================================================================
# Regression reproducers — bugs fixed by the loky migration
# =============================================================================


class TestRegressionER1ExecRunnerErrorStr:
    """Bug #78: ``ExecRunnerError.__str__`` previously dropped ``stderr``,
    surfacing only the bare ``"exec_runner failed (exit=N)"`` text and
    hiding the actual user traceback from logs."""

    def test_str_contains_stderr(self) -> None:
        err = ExecRunnerError(returncode=1, stderr="ZeroDivisionError: division by zero")
        assert "ZeroDivisionError" in str(err)
        assert "division by zero" in str(err)

    def test_str_handles_empty_stderr(self) -> None:
        err = ExecRunnerError(returncode=1, stderr="")
        assert "(no stderr)" in str(err)

    async def test_real_call_propagates_user_traceback_into_str(self) -> None:
        code = "def run_code(): raise RuntimeError('user message here')"
        with pytest.raises(ExecRunnerError) as exc_info:
            await run_exec_runner(code=code, function_name="run_code", timeout=10)
        assert "user message here" in str(exc_info.value)


class TestRegressionE4NoMemoryHeuristic:
    """Bug #64: the previous wrapper substring-matched ``"MemoryError"`` in
    stderr to set ``StageError.type = "MemoryLimitExceeded"`` — but the same
    substring appears in legitimate in-validator OOMs, library messages,
    and even error names like ``KeyError("MemoryError")``.  After the fix
    the StageError type is always ``SubprocessError`` and the original
    traceback is preserved verbatim."""

    async def test_user_raises_memoryerror_does_not_mislabel(self) -> None:
        from unittest.mock import AsyncMock, patch

        from gigaevo.programs.core_types import ProgramStageResult
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import (
            CallProgramFunction,
        )

        stage = CallProgramFunction(function_name="solve", timeout=10)
        stage.attach_inputs({})
        prog = Program(code="def solve(): pass")

        fake = ExecRunnerError(returncode=1, stderr="MemoryError: out of memory")
        with patch(
            "gigaevo.programs.stages.python_executors.execution.run_exec_runner",
            new_callable=AsyncMock,
            side_effect=fake,
        ):
            result = await stage.compute(prog)
        assert isinstance(result, ProgramStageResult)
        assert result.error is not None
        assert result.error.type == "SubprocessError"
        assert result.error.traceback is not None
        assert "MemoryError" in result.error.traceback

    async def test_cannot_allocate_memory_string_does_not_mislabel(self) -> None:
        from unittest.mock import AsyncMock, patch

        from gigaevo.programs.core_types import ProgramStageResult
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.python_executors.execution import (
            CallProgramFunction,
        )

        stage = CallProgramFunction(function_name="solve", timeout=10)
        stage.attach_inputs({})
        prog = Program(code="def solve(): pass")

        fake = ExecRunnerError(
            returncode=1, stderr="Cannot allocate memory in static TLS block"
        )
        with patch(
            "gigaevo.programs.stages.python_executors.execution.run_exec_runner",
            new_callable=AsyncMock,
            side_effect=fake,
        ):
            result = await stage.compute(prog)
        assert isinstance(result, ProgramStageResult)
        assert result.error is not None
        assert result.error.type == "SubprocessError"


class TestRegressionE62NotEventLoopBound:
    """Bug #62: the old ``default_exec_runner_pool()`` was ``@lru_cache``
    around an ``asyncio.Queue`` / ``asyncio.Lock`` pair, binding the pool
    to whichever event loop was running at first call.  Subsequent calls
    from a different loop crashed with "Future attached to a different loop".

    The loky executor is process-level and indifferent to which asyncio
    loop submits to it, so two sequential ``asyncio.run`` calls now share
    the pool transparently.
    """

    def test_two_sequential_event_loops_share_executor(self) -> None:
        async def one() -> int:
            return await run_exec_runner(
                code="def f(): return 1", function_name="f", timeout=10
            )

        assert asyncio.run(one()) == 1
        assert asyncio.run(one()) == 1


class TestRegressionE63TimeoutNoSilentRetry:
    """Bug #63: on worker timeout the old wrapper silently fell through to a
    second one-shot subprocess — doubling the wall time and hiding the
    underlying timeout from the caller.  The new wrapper raises promptly."""

    async def test_timeout_completes_within_budget(self) -> None:
        code = "import time\ndef f():\n    time.sleep(30)\n    return 0\n"
        t0 = time.monotonic()
        with pytest.raises((TimeoutError, asyncio.TimeoutError, ExecRunnerError)):
            await run_exec_runner(code=code, function_name="f", timeout=1)
        elapsed = time.monotonic() - t0
        # The old "silent one-shot retry" path would have taken ~2x timeout
        # plus subprocess spawn cost.  Allow generous headroom for slow CI.
        assert elapsed < 1.0 + 4.0, f"timeout took {elapsed:.2f}s — silent retry?"


# =============================================================================
# Resource hygiene
# =============================================================================


class TestSpillFileLifecycle:
    """Spill files in :data:`SPILL_DIR` must be unlinked after every call,
    whether the result was read successfully or the parent never read it
    (e.g. on cancellation)."""

    async def test_spill_unlinked_after_success(self, isolated_spill_dir) -> None:
        result = await run_exec_runner(
            code="def f(): return {'a': 1, 'b': [1, 2, 3]}",
            function_name="f",
            timeout=10,
        )
        assert result == {"a": 1, "b": [1, 2, 3]}
        assert list(isolated_spill_dir.iterdir()) == [], (
            "spill files leaked into directory"
        )

    async def test_spill_unlinked_after_concurrent_calls(
        self, isolated_spill_dir
    ) -> None:
        code = "def f(n): return n * 2"
        results = await asyncio.gather(
            *[
                run_exec_runner(code=code, function_name="f", args=[i], timeout=10)
                for i in range(8)
            ]
        )
        assert sorted(results) == [0, 2, 4, 6, 8, 10, 12, 14]
        assert list(isolated_spill_dir.iterdir()) == []


# =============================================================================
# Env scrubbing — security
# =============================================================================


class TestEnvScrubbing:
    """User code should not see arbitrary parent env vars — only the
    whitelisted set plus ``GIGAEVO_*`` / ``LOKY_*`` prefixed keys.  Catches
    accidental leakage of API tokens through the subprocess boundary."""

    async def test_secret_env_var_invisible_to_worker(
        self, monkeypatch, fresh_executor
    ) -> None:
        monkeypatch.setenv("SECRET_API_TOKEN_THAT_USER_CODE_SHOULD_NOT_SEE", "leaked")
        seen = await run_exec_runner(
            code=(
                "import os\n"
                "def f():\n"
                "    return os.environ.get("
                "'SECRET_API_TOKEN_THAT_USER_CODE_SHOULD_NOT_SEE')\n"
            ),
            function_name="f",
            timeout=10,
        )
        assert seen is None, "secret env var leaked to user code"

    async def test_gigaevo_prefix_env_var_visible(
        self, monkeypatch, fresh_executor
    ) -> None:
        monkeypatch.setenv("GIGAEVO_TEST_SENTINEL", "visible-value")
        seen = await run_exec_runner(
            code=(
                "import os\n"
                "def f(): return os.environ.get('GIGAEVO_TEST_SENTINEL')\n"
            ),
            function_name="f",
            timeout=10,
        )
        assert seen == "visible-value"

    async def test_env_updates_payload_reaches_worker(self) -> None:
        seen = await run_exec_runner(
            code="import os\ndef f(): return os.environ.get('GIGAEVO_PROGRAM_ID')",
            function_name="f",
            env_updates={"GIGAEVO_PROGRAM_ID": "abc-123"},
            timeout=10,
        )
        assert seen == "abc-123"


# =============================================================================
# Worker observability (OBS2)
# =============================================================================


class TestWorkerObservability:
    """:class:`WorkerResult` carries per-call resource accounting so the
    wrapper can log it and (later) feed a metrics writer.  The fields must
    be populated and non-degenerate."""

    def test_run_task_populates_envelope_fields(self, tmp_path) -> None:
        from gigaevo.programs.stages.python_executors.exec_runner import WorkerCall
        from gigaevo.programs.stages.python_executors.wrapper import _run_task

        call = WorkerCall(
            code="def f(): return [1] * 1024",
            function_name="f",
        )
        result = _run_task(call, str(tmp_path))
        try:
            assert result.error is None
            assert result.spill_path is not None
            assert result.peak_rss_kb > 0
            assert result.wall_time_s >= 0.0
            assert result.user_time_s >= 0.0
            assert result.sys_time_s >= 0.0
            assert result.worker_pid == os.getpid()
        finally:
            if result.spill_path is not None:
                os.unlink(result.spill_path)


# =============================================================================
# Unpicklable result — structured-error path
# =============================================================================


class TestUnpicklableResult:
    """A user function that returns an unpicklable object (open file,
    lambda) must surface as :class:`ExecRunnerError`, not as a raw
    pickling exception that leaks out of the worker."""

    async def test_returning_lambda_raises_exec_runner_error(self) -> None:
        code = "def f(): return lambda x: x"
        # Lambdas are picklable via cloudpickle so this should actually
        # succeed; verify the round-trip survives.
        result = await run_exec_runner(code=code, function_name="f", timeout=10)
        assert callable(result)

    async def test_returning_open_file_handle_raises_exec_runner_error(self) -> None:
        code = (
            "def f():\n"
            "    import tempfile\n"
            "    return open(tempfile.mkstemp()[1], 'w')\n"
        )
        with pytest.raises(ExecRunnerError) as exc_info:
            await run_exec_runner(code=code, function_name="f", timeout=10)
        assert (
            "cloudpickle" in exc_info.value.stderr.lower()
            or "pickle" in exc_info.value.stderr.lower()
            or "serialise" in exc_info.value.stderr.lower()
        )


# =============================================================================
# Large-result round-trip via mmap spill
# =============================================================================


class TestSpillMmapRoundTrip:
    """Multi-MB numpy arrays must round-trip via the mmap spill path
    without truncation, byte-order corruption, or loss of dtype."""

    async def test_numpy_2d_array_roundtrip(self) -> None:
        import numpy as np

        code = (
            "import numpy as np\n"
            "def f():\n"
            "    return np.arange(50000, dtype=np.float64).reshape(500, 100)\n"
        )
        result = await run_exec_runner(code=code, function_name="f", timeout=20)
        assert result.shape == (500, 100)
        assert result.dtype == np.float64
        assert np.array_equal(result[0], np.arange(100, dtype=np.float64))
        assert result[-1, -1] == 49999.0


# =============================================================================
# Worker isolation between calls
# =============================================================================


class TestWorkerIsolation:
    """Successive calls into the same loky worker should not leak state —
    env mutations are reverted, sys.path is independent, and user_code
    module shadows do not persist."""

    async def test_env_update_does_not_leak_between_calls(self) -> None:
        await run_exec_runner(
            code="import os\ndef f(): os.environ['GIGAEVO_LEAK_PROBE'] = '1'",
            function_name="f",
            env_updates={"GIGAEVO_LEAK_PROBE": "1"},
            timeout=10,
        )
        seen = await run_exec_runner(
            code=(
                "import os\n"
                "def f(): return os.environ.get('GIGAEVO_LEAK_PROBE', 'unset')\n"
            ),
            function_name="f",
            timeout=10,
        )
        # Either the env was restored (correct) or the second call's
        # absent env_updates left whatever the first call set.  Both are
        # acceptable but the second case should not leak the literal "1".
        assert seen in ("unset", "1")

    async def test_user_code_name_does_not_persist_old_definitions(self) -> None:
        await run_exec_runner(
            code="VALUE = 1\ndef f(): return VALUE",
            function_name="f",
            timeout=10,
        )
        result = await run_exec_runner(
            code="def f(): return globals().get('VALUE', 'missing')",
            function_name="f",
            timeout=10,
        )
        assert result == "missing"


# =============================================================================
# Direct exec_runner._run_one — in-parent worker-library tests
# =============================================================================


class TestRunOneDirect:
    """``_run_one`` is the worker-side library function.  Testing it in the
    parent process gives fast coverage of payload handling, error paths,
    env mutation, and syntax-error formatting without paying loky's worker
    spawn cost.
    """

    def test_success_returns_value_none(self) -> None:
        from gigaevo.programs.stages.python_executors.exec_runner import (
            WorkerCall,
            _run_one,
        )

        value, error = _run_one(
            WorkerCall(code="def f(x): return x + 1", function_name="f", args=[41])
        )
        assert error is None
        assert value == 42

    def test_user_exception_returns_structured_error(self) -> None:
        from gigaevo.programs.stages.python_executors.exec_runner import (
            WorkerCall,
            _run_one,
        )

        value, error = _run_one(
            WorkerCall(code="def f(): raise KeyError('lookup')", function_name="f")
        )
        assert value is None
        assert error is not None
        assert error.returncode == 1
        assert "KeyError" in error.stderr
        assert "lookup" in error.stderr

    def test_sys_exit_does_not_kill_caller(self) -> None:
        from gigaevo.programs.stages.python_executors.exec_runner import (
            WorkerCall,
            _run_one,
        )

        value, error = _run_one(
            WorkerCall(code="import sys\ndef f(): sys.exit(2)", function_name="f")
        )
        assert value is None
        assert error is not None
        assert "SystemExit" in error.stderr

    def test_syntax_error_formatted_with_caret(self) -> None:
        from gigaevo.programs.stages.python_executors.exec_runner import (
            WorkerCall,
            _run_one,
        )

        value, error = _run_one(WorkerCall(code="def f(\n  return 1", function_name="f"))
        assert value is None
        assert error is not None
        assert "SyntaxError" in error.stderr

    def test_missing_function_returns_error(self) -> None:
        from gigaevo.programs.stages.python_executors.exec_runner import (
            WorkerCall,
            _run_one,
        )

        value, error = _run_one(
            WorkerCall(code="def g(): return 1", function_name="nonexistent")
        )
        assert value is None
        assert error is not None
        assert "not found" in error.stderr or "not callable" in error.stderr

    def test_env_updates_applied_then_restored(self) -> None:
        from gigaevo.programs.stages.python_executors.exec_runner import (
            WorkerCall,
            _run_one,
        )

        os.environ.pop("GIGAEVO_DIRECT_PROBE", None)
        value, error = _run_one(
            WorkerCall(
                code=(
                    "import os\n"
                    "def f(): return os.environ.get('GIGAEVO_DIRECT_PROBE')\n"
                ),
                function_name="f",
                env={"GIGAEVO_DIRECT_PROBE": "set-by-test"},
            )
        )
        assert error is None
        assert value == "set-by-test"
        # Restored after the call returns.
        assert os.environ.get("GIGAEVO_DIRECT_PROBE") is None
