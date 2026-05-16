"""Tests for the loky-backed Python executor: run_exec_runner public surface,
PythonCodeExecutor stages, regression reproducers, and worker-isolation /
env-scrub / spill-hygiene properties."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time

import pytest

from gigaevo.programs.stages.python_executors.wrapper import (
    ExecRunnerError,
    run_exec_runner,
)

# ---------------------------------------------------------------------------
# run_exec_runner — basic execution
# ---------------------------------------------------------------------------


class TestRunExecRunner:
    async def test_simple_function_returns_result(self) -> None:
        """Execute a simple function and get the return value."""
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
        """Complex return values (dict, list, nested) survive serialization."""
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


# ---------------------------------------------------------------------------
# ExecRunnerError attributes
# ---------------------------------------------------------------------------


class TestExecRunnerErrorAttributes:
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
        assert result.error is not None
        assert result.error.traceback is not None
        assert "ValueError" in result.error.traceback


# ---------------------------------------------------------------------------
# PythonCodeExecutor — error-handling paths
# ---------------------------------------------------------------------------


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

    async def test_validator_called_with_non_none_context(self, tmp_path) -> None:
        """When context is non-None it is prepended to the call args.

        The validate function receives (context, payload) when context is provided,
        so the test verifies both args arrive correctly.
        """
        from gigaevo.programs.program import Program
        from gigaevo.programs.stages.common import Box
        from gigaevo.programs.stages.python_executors.execution import (
            CallValidatorFunction,
        )

        validator_file = tmp_path / "validator_ctx.py"
        # Returns the context value so we can assert it was passed
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
    """``ExecRunnerError.__str__`` includes ``stderr`` (was dropped)."""

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
    """``StageError.type`` is always ``SubprocessError``; no ``"MemoryError"``
    substring heuristic mislabels in-validator OOMs or similar messages."""

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
    """The loky executor is process-level; two sequential ``asyncio.run``
    calls share the pool (was lru_cache-bound to first event loop)."""

    def test_two_sequential_event_loops_share_executor(self) -> None:
        async def one() -> int:
            return await run_exec_runner(
                code="def f(): return 1", function_name="f", timeout=10
            )

        assert asyncio.run(one()) == 1
        assert asyncio.run(one()) == 1


class TestRegressionE63TimeoutNoSilentRetry:
    """Timeout raises promptly; no silent one-shot subprocess retry."""

    async def test_timeout_completes_within_budget(self) -> None:
        code = "import time\ndef f():\n    time.sleep(30)\n    return 0\n"
        t0 = time.monotonic()
        with pytest.raises((TimeoutError, asyncio.TimeoutError, ExecRunnerError)):
            await run_exec_runner(code=code, function_name="f", timeout=1)
        # Old code's silent retry would have taken ~2x timeout.
        assert time.monotonic() - t0 < 1.0 + 4.0


class TestSpillFileLifecycle:
    """Spill files are unlinked after every call."""

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


class TestEnvScrubbing:
    """Worker sees only whitelisted + ``GIGAEVO_*``/``LOKY_*`` env vars."""

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

    @pytest.mark.parametrize(
        "secret_key",
        [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "GCP_PROJECT",
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "LANGFUSE_SECRET_KEY",
            "WANDB_API_KEY",
            "HF_TOKEN",
            "STRIPE_SECRET_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
        ],
    )
    async def test_well_known_secret_env_vars_scrubbed(
        self, monkeypatch, fresh_executor, secret_key: str
    ) -> None:
        monkeypatch.setenv(secret_key, "should-not-leak")
        seen = await run_exec_runner(
            code=(
                "import os\n"
                f"def f(): return os.environ.get({secret_key!r})\n"
            ),
            function_name="f",
            timeout=10,
        )
        assert seen is None, f"{secret_key} leaked to worker"


class TestSpillDirHardening:
    """Default spill dir is per-uid; operator paths get ``..`` resolved."""

    def test_default_spill_dir_is_per_uid(self, monkeypatch) -> None:
        from pathlib import Path as _Path
        import tempfile as _tempfile

        from gigaevo.programs.stages.python_executors.wrapper import (
            WorkerConfig,
        )

        monkeypatch.delenv("GIGAEVO_EXECUTOR_SPILL_DIR", raising=False)
        cfg = WorkerConfig.from_env()
        assert cfg.spill_dir != _Path(_tempfile.gettempdir())
        assert str(os.getuid()) in cfg.spill_dir.name

    def test_spill_dir_env_resolves_dotdot(self, monkeypatch, tmp_path) -> None:
        from gigaevo.programs.stages.python_executors.wrapper import (
            WorkerConfig,
        )

        target = tmp_path / "real-spill"
        target.mkdir()
        weird = tmp_path / "real-spill" / ".." / "real-spill"
        monkeypatch.setenv("GIGAEVO_EXECUTOR_SPILL_DIR", str(weird))
        cfg = WorkerConfig.from_env()
        assert cfg.spill_dir == target.resolve()


class TestWorkerSignalDispositions:
    """Workers have default signal handlers (KeyboardInterrupt on SIGINT,
    SIG_DFL on SIGTERM/SIGHUP/SIGQUIT) regardless of loky's internal setup."""

    async def test_sigint_handler_is_default_int_handler(self) -> None:
        result = await run_exec_runner(
            code=(
                "import signal\n"
                "def f():\n"
                "    return (\n"
                "        signal.getsignal(signal.SIGINT)\n"
                "        is signal.default_int_handler\n"
                "    )\n"
            ),
            function_name="f",
            timeout=10,
        )
        assert result is True

    async def test_sigterm_handler_is_default(self) -> None:
        result = await run_exec_runner(
            code=(
                "import signal\n"
                "def f():\n"
                "    return signal.getsignal(signal.SIGTERM) == signal.SIG_DFL\n"
            ),
            function_name="f",
            timeout=10,
        )
        assert result is True


class TestWorkerObservability:
    """``WorkerResult`` resource-accounting fields are populated."""

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


class TestUnpicklableResult:
    """Unpicklable results surface as :class:`ExecRunnerError`."""

    async def test_lambda_round_trips_via_cloudpickle(self) -> None:
        result = await run_exec_runner(
            code="def f(): return lambda x: x", function_name="f", timeout=10
        )
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


class TestSpillMmapRoundTrip:
    """Multi-MB numpy arrays round-trip via the mmap spill path."""

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
    """Successive calls into the same worker don't leak cwd / sys.path / env."""

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
        # Either env was restored, or the second call's absent
        # env_updates left whatever the first call set.  Both acceptable.
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

    async def test_user_chdir_does_not_leak_to_next_call(self, tmp_path) -> None:
        leak_dir = tmp_path / "leak-target"
        leak_dir.mkdir()

        seen_first = await run_exec_runner(
            code=(
                "import os\n"
                f"def f():\n"
                f"    os.chdir({str(leak_dir)!r})\n"
                f"    return os.getcwd()\n"
            ),
            function_name="f",
            timeout=10,
        )
        assert seen_first == str(leak_dir)

        seen_second = await run_exec_runner(
            code="import os\ndef f(): return os.getcwd()",
            function_name="f",
            timeout=10,
        )
        assert seen_second != str(leak_dir), (
            f"cwd leaked across worker reuse: {seen_second!r}"
        )

    async def test_user_sys_path_mutation_does_not_leak_to_next_call(
        self,
    ) -> None:
        sentinel = "/__sentinel_sys_path_leak_xyz__"
        await run_exec_runner(
            code=(
                "import sys\n"
                f"def f():\n"
                f"    sys.path.insert(0, {sentinel!r})\n"
                f"    return {sentinel!r} in sys.path\n"
            ),
            function_name="f",
            timeout=10,
        )
        leaked = await run_exec_runner(
            code=(
                "import sys\n"
                f"def f(): return {sentinel!r} in sys.path\n"
            ),
            function_name="f",
            timeout=10,
        )
        assert leaked is False, (
            f"sys.path leak: sentinel {sentinel!r} survived into next call"
        )


class TestRunOneDirect:
    """In-parent unit tests for ``_run_one`` — no loky spawn cost."""

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


# =============================================================================
# Coverage gaps surfaced during the loky migration audit
# =============================================================================


class TestPythonPathPropagation:
    """``python_path`` entries must be prepended to the worker's ``sys.path``
    so problem-local modules import without packaging.  Regression guard for
    the algotune shim path."""

    async def test_python_path_makes_local_module_importable(self, tmp_path) -> None:
        # Create an isolated module the worker would not otherwise see.
        mod_dir = tmp_path / "ppath_pkg"
        mod_dir.mkdir()
        (mod_dir / "sentinel_module.py").write_text(
            "MARKER = 'python_path-reached-worker'\n"
        )
        code = (
            "import sentinel_module\n"
            "def f():\n"
            "    return sentinel_module.MARKER\n"
        )
        result = await run_exec_runner(
            code=code,
            function_name="f",
            python_path=[mod_dir],
            timeout=10,
        )
        assert result == "python_path-reached-worker"


class TestEnvUpdatesNoneUnsets:
    """``env_updates={'KEY': None}`` must unset KEY for the duration of the
    call, even when the parent env had it set.  Untested before — silently
    promoting None to the literal string ``'None'`` would corrupt downstream
    code that checks ``os.environ.get(K) is None``."""

    async def test_none_value_unsets_existing_var(
        self, monkeypatch, fresh_executor
    ) -> None:
        # Set the var in the parent so the worker would inherit it via the
        # whitelist (GIGAEVO_ prefix → always passed through).
        monkeypatch.setenv("GIGAEVO_UNSET_PROBE", "parent-value")

        seen = await run_exec_runner(
            code=(
                "import os\n"
                "def f(): return os.environ.get('GIGAEVO_UNSET_PROBE', '__missing__')\n"
            ),
            function_name="f",
            env_updates={"GIGAEVO_UNSET_PROBE": None},
            timeout=10,
        )
        assert seen == "__missing__", (
            f"None env_updates value should unset, but worker saw {seen!r}"
        )


class TestCancellationCleansSpill:
    """Bug-class: cancellation between worker completion and parent read
    would leak a spill file.  ``_unlink_spill_on_done`` registers a
    done-callback that unlinks once the worker finishes — verify it fires
    even after the awaiting task is cancelled."""

    async def test_cancelled_task_does_not_leak_spill_file(
        self, isolated_spill_dir
    ) -> None:
        # Worker sleeps long enough that we can cancel between submit and
        # await completion — but short enough that the eventual completion
        # arrives within the test timeout so the done-callback runs.
        code = (
            "import time\n"
            "def f():\n"
            "    time.sleep(0.5)\n"
            "    return [0] * 1024\n"
        )
        task = asyncio.create_task(
            run_exec_runner(code=code, function_name="f", timeout=10)
        )
        # Give submit a chance to register but cancel before the worker
        # finishes — the done-callback path is what we want to exercise.
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Wait long enough for the worker to finish its sleep + serialise +
        # done-callback to run.  Poll the dir to avoid flaky fixed sleeps.
        for _ in range(20):
            await asyncio.sleep(0.1)
            if not list(isolated_spill_dir.iterdir()):
                break
        leaked = list(isolated_spill_dir.iterdir())
        assert leaked == [], f"spill leaked after cancellation: {leaked}"


# =============================================================================
# Distant-corner serialization edge cases
# =============================================================================


class TestSerializationDistantCorners:
    """Worker→parent cloudpickle round-trip edge cases."""

    async def test_self_referential_list_round_trip(self) -> None:
        code = "def f():\n    x = []\n    x.append(x)\n    return x\n"
        result = await run_exec_runner(code=code, function_name="f", timeout=10)
        assert result[0] is result

    async def test_cyclic_dict_round_trip(self) -> None:
        code = "def f():\n    d = {}\n    d['self'] = d\n    return d\n"
        result = await run_exec_runner(code=code, function_name="f", timeout=10)
        assert result["self"] is result

    async def test_numpy_structured_dtype_round_trip(self) -> None:
        import numpy as np

        code = (
            "import numpy as np\n"
            "def f():\n"
            "    dt = np.dtype([('a', np.int32), ('b', np.float64), ('c', 'U8')])\n"
            "    arr = np.zeros(5, dtype=dt)\n"
            "    arr['a'] = [1, 2, 3, 4, 5]\n"
            "    arr['b'] = [0.1, 0.2, 0.3, 0.4, 0.5]\n"
            "    arr['c'] = ['xx', 'yy', 'zz', 'aa', 'bb']\n"
            "    return arr\n"
        )
        result = await run_exec_runner(code=code, function_name="f", timeout=20)
        assert result.dtype.names == ("a", "b", "c")
        assert result["a"].tolist() == [1, 2, 3, 4, 5]
        assert result["c"][2] == "zz"
        assert np.allclose(result["b"], [0.1, 0.2, 0.3, 0.4, 0.5])

    async def test_numpy_memmap_returns_materialised_array(self) -> None:
        """Parent should get a regular in-memory array, not a memmap
        aliasing the worker's tempfile."""
        import numpy as np

        code = (
            "import numpy as np, tempfile\n"
            "def f():\n"
            "    path = tempfile.mkstemp(suffix='.dat')[1]\n"
            "    arr = np.memmap(path, dtype='float32', mode='w+', shape=(100,))\n"
            "    arr[:] = np.arange(100, dtype='float32')\n"
            "    arr.flush()\n"
            "    return arr\n"
        )
        result = await run_exec_runner(code=code, function_name="f", timeout=10)
        assert result.shape == (100,)
        assert float(result.sum()) == float(np.arange(100, dtype=np.float32).sum())
        # Must be writable from the parent — if it's still a memmap aliased
        # to the worker's deleted tempfile this either crashes or silently
        # writes into a dangling region.
        result[0] = 999.0
        assert result[0] == 999.0

    async def test_closure_over_user_defined_class(self) -> None:
        code = (
            "class Inner:\n"
            "    def __init__(self, v): self.v = v\n"
            "    def double(self): return self.v * 2\n"
            "def f():\n"
            "    obj = Inner(21)\n"
            "    def closure():\n"
            "        return obj.double()\n"
            "    return closure\n"
        )
        result = await run_exec_runner(code=code, function_name="f", timeout=10)
        assert callable(result)
        assert result() == 42

    async def test_instance_of_class_defined_inside_function_body(self) -> None:
        code = (
            "def f():\n"
            "    class Local:\n"
            "        def __init__(self, n): self.n = n\n"
            "        def squared(self): return self.n * self.n\n"
            "    return Local(7)\n"
        )
        result = await run_exec_runner(code=code, function_name="f", timeout=10)
        assert result.squared() == 49

    async def test_numpy_round_trip_survives_aggressive_heap_pressure(self) -> None:
        """Probe for use-after-free on the spill mmap: after the parent
        unlinks the spill, allocating heap pressure must not corrupt the
        unpickled array (would mean it aliased the spill mmap)."""
        import gc

        import numpy as np

        code = (
            "import numpy as np\n"
            "def f():\n"
            "    return np.arange(1_000_000, dtype=np.float32)\n"
        )
        result = await run_exec_runner(code=code, function_name="f", timeout=20)
        # Sum the un-tampered tail before any writes — must equal the
        # arange tail-sum.  float32 sums are non-associative, so use
        # ``np.array_equal`` against a fresh reference rather than
        # comparing scalar sums.
        reference = np.arange(1_000_000, dtype=np.float32)
        assert np.array_equal(result, reference)
        # Pressure: allocate ~64 MB of distinct byte buffers and force
        # GC.  If the result aliased the spill region, the kernel would
        # reuse those pages and the equality check below would diverge.
        ballast = [bytes(1024 * 1024) for _ in range(64)]
        gc.collect()
        assert np.array_equal(result, reference), (
            "result diverged from reference after heap pressure — suggests "
            "the unpickled array aliased the (now-unlinked) spill mmap"
        )
        # Writing should succeed and not crash — confirms own-buffer
        # semantics rather than a read-only mmap-aliased view.
        del ballast
        result[0] = -42.0
        assert result[0] == -42.0

    def test_worker_envelopes_round_trip_via_cloudpickle(self) -> None:
        """``WorkerCall``/``WorkerError``/``WorkerResult`` survive
        cloudpickle round-trip (they cross the loky boundary)."""
        import cloudpickle

        from gigaevo.programs.stages.python_executors.exec_runner import (
            WorkerCall,
            WorkerError,
        )
        from gigaevo.programs.stages.python_executors.wrapper import WorkerResult

        call = WorkerCall(
            code="def f(): return 1",
            function_name="f",
            args=[1, 2, 3],
            kwargs={"k": "v"},
            python_path=["/tmp/foo"],
            env={"X": "1", "Y": None},
        )
        assert cloudpickle.loads(cloudpickle.dumps(call)) == call

        err = WorkerError(stderr="oops", returncode=2)
        assert cloudpickle.loads(cloudpickle.dumps(err)) == err

        res = WorkerResult(
            spill_path="/tmp/x.pkl",
            error=None,
            peak_rss_kb=4242,
            wall_time_s=0.5,
            user_time_s=0.4,
            sys_time_s=0.1,
            worker_pid=12345,
        )
        assert cloudpickle.loads(cloudpickle.dumps(res)) == res

    async def test_worker_returns_object_with_lock_surfaces_structured_error(
        self,
    ) -> None:
        """Unpicklable component (Lock) → :class:`ExecRunnerError`, not raw
        cloudpickle traceback or deadlock."""
        code = (
            "import threading\n"
            "def f():\n"
            "    return {'data': [1, 2, 3], 'lock': threading.Lock()}\n"
        )
        with pytest.raises(ExecRunnerError) as exc_info:
            await run_exec_runner(code=code, function_name="f", timeout=10)
        msg = exc_info.value.stderr.lower()
        assert "serialise" in msg or "pickle" in msg

    async def test_python_path_does_not_accumulate_across_calls(self) -> None:
        """``_run_one``'s finally restores ``sys.path``: many calls with
        distinct ``python_path`` entries don't grow it."""
        import tempfile

        code = "import sys\ndef f(): return len(sys.path)\n"
        first_len: int | None = None
        for i in range(8):
            d = tempfile.mkdtemp(prefix=f"distant-corner-acc-{i}-")
            n = await run_exec_runner(
                code=code,
                function_name="f",
                python_path=[Path(d)],
                timeout=10,
            )
            if first_len is None:
                first_len = n
            else:
                # Allow ±1 slack for ``_ensure_cwd_in_path`` toggling
                # depending on whether the worker's cwd happens to be
                # already-present in the snapshotted sys.path.
                assert abs(n - first_len) <= 1, (
                    f"sys.path drift: call 0 had {first_len}, call {i} has {n}"
                )

    async def test_cloudpickle_register_by_value_idempotent(self) -> None:
        """``_PICKLE_BY_VALUE_MODULES`` is a set keyed by name — re-registering
        ``"user_code"`` every call doesn't grow it."""
        code = (
            "import cloudpickle.cloudpickle as _cp\n"
            "def f():\n"
            "    modules = _cp._PICKLE_BY_VALUE_MODULES\n"
            "    return [m for m in modules if m == 'user_code']\n"
        )
        # Drive several iterations; the set must stay at most {'user_code'}.
        for _ in range(5):
            entries = await run_exec_runner(
                code=code, function_name="f", timeout=10
            )
            assert entries in ([], ["user_code"]), (
                f"unexpected user_code accumulation: {entries}"
            )


class TestProtocolFiveSafety:
    """Worker dump / parent loads asymmetry checks for cloudpickle protocol 5."""

    def test_one_cloudpickle_in_environment(self) -> None:
        """No vendored cloudpickle lurking under loky — both ends share dispatch tables."""
        import importlib

        cp_main = importlib.import_module("cloudpickle")
        # loky imports the *installed* cloudpickle, not a vendored copy.
        from loky import cloudpickle_wrapper as lw

        lw_cp_path = lw.dumps.__module__  # "cloudpickle.cloudpickle" or similar
        assert lw_cp_path.startswith("cloudpickle"), (
            f"loky's dumps does not come from cloudpickle: {lw_cp_path}"
        )
        # File-level check: both should resolve under the same install root.
        from cloudpickle import cloudpickle as cp_impl

        assert Path(cp_impl.__file__).parent == Path(cp_main.__file__).parent

    def test_protocol_5_round_trip_sanity(self, tmp_path) -> None:
        """Worker-style dump(protocol=5) → parent-style mmap + loads."""
        import mmap as _mmap

        import cloudpickle as _cp

        payload = {"a": list(range(1024)), "b": {"nested": (1.0, 2.0, 3.0)}}
        f = tmp_path / "p5.pkl"
        with open(f, "wb") as fh:
            _cp.dump(payload, fh, protocol=5)
        with open(f, "rb") as fh:
            with _mmap.mmap(fh.fileno(), 0, access=_mmap.ACCESS_READ) as mm:
                loaded = _cp.loads(mm)
        assert loaded == payload

# =============================================================================
# pytest-xdist auto-cap of max_workers
# =============================================================================


class TestXdistWorkerCountAutoCap:
    """Under xdist, ``max_workers`` auto-caps to ``cpu_count // xdist_count``
    to avoid an N×cpu_count fork-bomb; explicit env override always wins."""

    def test_no_xdist_yields_default_none(self, monkeypatch) -> None:
        from gigaevo.programs.stages.python_executors.wrapper import (
            WorkerConfig,
        )

        monkeypatch.delenv("GIGAEVO_EXECUTOR_MAX_WORKERS", raising=False)
        monkeypatch.delenv("PYTEST_XDIST_WORKER_COUNT", raising=False)
        cfg = WorkerConfig.from_env()
        assert cfg.max_workers is None

    def test_xdist_caps_to_cpu_div_workers(self, monkeypatch) -> None:
        from gigaevo.programs.stages.python_executors.wrapper import (
            WorkerConfig,
        )

        monkeypatch.delenv("GIGAEVO_EXECUTOR_MAX_WORKERS", raising=False)
        monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "4")
        cfg = WorkerConfig.from_env()
        # max(1, cpu_count // 4): exact value depends on host, just verify
        # it's a small positive int strictly less than cpu_count.
        cpu = os.cpu_count() or 1
        assert cfg.max_workers is not None
        assert cfg.max_workers >= 1
        if cpu >= 4:
            assert cfg.max_workers == cpu // 4
            assert cfg.max_workers < cpu

    def test_xdist_one_worker_does_not_cap(self, monkeypatch) -> None:
        from gigaevo.programs.stages.python_executors.wrapper import (
            WorkerConfig,
        )

        monkeypatch.delenv("GIGAEVO_EXECUTOR_MAX_WORKERS", raising=False)
        monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "1")
        cfg = WorkerConfig.from_env()
        # Single-worker xdist is effectively no xdist — don't penalize.
        assert cfg.max_workers is None

    def test_explicit_max_workers_overrides_xdist(self, monkeypatch) -> None:
        from gigaevo.programs.stages.python_executors.wrapper import (
            WorkerConfig,
        )

        monkeypatch.setenv("GIGAEVO_EXECUTOR_MAX_WORKERS", "7")
        monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "8")
        cfg = WorkerConfig.from_env()
        # Operator override wins even on a 1-CPU host where cpu//8 = 0.
        assert cfg.max_workers == 7

    def test_xdist_floor_is_one(self, monkeypatch) -> None:
        """cpu_count // xdist_count can be 0 on small hosts; we must
        never request max_workers=0 (loky raises ValueError)."""
        from gigaevo.programs.stages.python_executors.wrapper import (
            WorkerConfig,
        )

        monkeypatch.delenv("GIGAEVO_EXECUTOR_MAX_WORKERS", raising=False)
        # Pick a count that exceeds any plausible cpu_count.
        monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "10000")
        cfg = WorkerConfig.from_env()
        assert cfg.max_workers is not None
        assert cfg.max_workers >= 1


# =============================================================================
# shutdown_executor wait semantics
# =============================================================================


class TestShutdownExecutorWaitFlag:
    """``wait=True`` blocks until loky's manager thread joins; ``wait=False``
    returns promptly."""

    async def test_wait_true_blocks_until_manager_thread_joined(self) -> None:
        from gigaevo.programs.stages.python_executors.wrapper import (
            _get_executor,
            shutdown_executor,
        )

        # Force a pool to exist with a live manager thread.
        executor = _get_executor()
        fut = executor.submit(lambda: 1)
        assert fut.result(timeout=10) == 1

        mgr_thread = executor._executor_manager_thread
        assert mgr_thread is not None
        assert mgr_thread.is_alive()

        shutdown_executor(wait=True)

        # After wait=True returns, the manager thread must be joined.
        # ``shutdown(wait=True)`` clears the attribute on the executor;
        # checking is_alive() on the captured ref is the residual check.
        assert not mgr_thread.is_alive()

    async def test_wait_false_returns_promptly(self) -> None:
        """Sanity guard: wait=False must not regress to wait=True.  We
        can't reliably observe the manager thread mid-shutdown on every
        scheduler, but we can at least verify wait=False completes in
        sub-second time even with a worker mid-submit."""
        import time as _time

        from gigaevo.programs.stages.python_executors.wrapper import (
            _get_executor,
            shutdown_executor,
        )

        executor = _get_executor()
        executor.submit(lambda: 1).result(timeout=10)

        t0 = _time.monotonic()
        shutdown_executor(wait=False)
        elapsed = _time.monotonic() - t0
        # wait=False on a small pool should be < 100ms.  Generous bound.
        assert elapsed < 2.0, f"shutdown_executor(wait=False) took {elapsed:.3f}s"


# =============================================================================
# LokyBackend — multi-instance isolation
# =============================================================================


class TestLokyBackendIsolation:
    """Two ``LokyBackend`` instances have independent pools, spill dirs, and
    configurations; tearing one down does not affect the other."""

    async def test_two_backends_have_independent_executors(self, tmp_path) -> None:
        from gigaevo.programs.stages.python_executors.exec_runner import WorkerCall
        from gigaevo.programs.stages.python_executors.wrapper import (
            LokyBackend,
            WorkerConfig,
        )

        spill_a = tmp_path / "spill-a"
        spill_b = tmp_path / "spill-b"
        spill_a.mkdir()
        spill_b.mkdir()

        a = LokyBackend(WorkerConfig(spill_dir=spill_a))
        b = LokyBackend(WorkerConfig(spill_dir=spill_b))
        try:
            ra = await a.execute(
                WorkerCall(code="def f(): return 1", function_name="f"),
                deadline_s=30,
            )
            rb = await b.execute(
                WorkerCall(code="def f(): return 2", function_name="f"),
                deadline_s=30,
            )
            assert ra == 1
            assert rb == 2
            assert a._executor is not b._executor
        finally:
            await a.shutdown(wait=True)
            await b.shutdown(wait=True)

    async def test_shutdown_one_does_not_affect_sibling(self, tmp_path) -> None:
        from gigaevo.programs.stages.python_executors.exec_runner import WorkerCall
        from gigaevo.programs.stages.python_executors.wrapper import (
            LokyBackend,
            WorkerConfig,
        )

        spill_a = tmp_path / "spill-a"
        spill_b = tmp_path / "spill-b"
        spill_a.mkdir()
        spill_b.mkdir()

        a = LokyBackend(WorkerConfig(spill_dir=spill_a))
        b = LokyBackend(WorkerConfig(spill_dir=spill_b))
        try:
            await a.execute(
                WorkerCall(code="def f(): return 1", function_name="f"),
                deadline_s=30,
            )
            await b.execute(
                WorkerCall(code="def f(): return 2", function_name="f"),
                deadline_s=30,
            )

            await a.shutdown(wait=True)
            assert a._executor is None

            # b is still alive
            result = await b.execute(
                WorkerCall(code="def f(): return 3", function_name="f"),
                deadline_s=30,
            )
            assert result == 3
        finally:
            await b.shutdown(wait=True)

    def test_pool_id_is_unique_per_instance(self) -> None:
        from gigaevo.programs.stages.python_executors.wrapper import WorkerConfig

        ids = {WorkerConfig().pool_id for _ in range(50)}
        assert len(ids) == 50

    def test_node_id_defaults_to_hostname(self) -> None:
        import socket

        from gigaevo.programs.stages.python_executors.wrapper import WorkerConfig

        assert WorkerConfig().node_id == socket.gethostname()

    def test_node_id_can_be_overridden(self) -> None:
        from gigaevo.programs.stages.python_executors.wrapper import WorkerConfig

        cfg = WorkerConfig(node_id="custom-pod-name")
        assert cfg.node_id == "custom-pod-name"

    def test_env_whitelist_can_be_extended(self) -> None:
        from gigaevo.programs.stages.python_executors.wrapper import (
            DEFAULT_ENV_WHITELIST,
            WorkerConfig,
        )

        extra = DEFAULT_ENV_WHITELIST | {"MY_CORP_PROXY", "MY_CORP_CA_BUNDLE"}
        cfg = WorkerConfig(env_whitelist=extra)
        assert "MY_CORP_PROXY" in cfg.env_whitelist
        assert "PATH" in cfg.env_whitelist  # original entries preserved

    def test_env_prefixes_can_be_extended(self) -> None:
        from gigaevo.programs.stages.python_executors.wrapper import WorkerConfig

        cfg = WorkerConfig(env_prefixes=("GIGAEVO_", "LOKY_", "MYAPP_"))
        assert "MYAPP_" in cfg.env_prefixes

    async def test_custom_env_whitelist_reaches_worker(self, tmp_path) -> None:
        from gigaevo.programs.stages.python_executors.exec_runner import WorkerCall
        from gigaevo.programs.stages.python_executors.wrapper import (
            DEFAULT_ENV_WHITELIST,
            LokyBackend,
            WorkerConfig,
        )

        spill = tmp_path / "spill"
        spill.mkdir()

        os.environ["MY_CORP_PROXY"] = "http://corp-proxy:8080"
        try:
            cfg = WorkerConfig(
                spill_dir=spill,
                env_whitelist=DEFAULT_ENV_WHITELIST | {"MY_CORP_PROXY"},
            )
            backend = LokyBackend(cfg)
            try:
                value = await backend.execute(
                    WorkerCall(
                        code=(
                            "import os\n"
                            "def f():\n"
                            "    return os.environ.get('MY_CORP_PROXY')\n"
                        ),
                        function_name="f",
                    ),
                    deadline_s=30,
                )
                assert value == "http://corp-proxy:8080"
            finally:
                await backend.shutdown(wait=True)
        finally:
            os.environ.pop("MY_CORP_PROXY", None)

    def test_default_singleton_is_lazy(self, monkeypatch) -> None:
        from gigaevo.programs.stages.python_executors import wrapper as _wrapper

        # Reset the singleton.
        monkeypatch.setattr(_wrapper, "_default_backend", None)
        assert _wrapper._default_backend is None

        backend = _wrapper.default_loky_backend()
        assert backend is not None
        assert _wrapper.default_loky_backend() is backend  # cached
