"""Tests for Stage.execute() — return type dispatch, error handling, caching, and cleanup."""

from __future__ import annotations

import asyncio

from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageIO,
    StageState,
    VoidInput,
    VoidOutput,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE

# ---------------------------------------------------------------------------
# Custom StageIO for tests
# ---------------------------------------------------------------------------


class MockOutput(StageIO):
    value: int = 42


# ---------------------------------------------------------------------------
# Stage subclasses exercising every return path
# ---------------------------------------------------------------------------


class ReturnOutputStage(Stage):
    """compute() returns an OutputModel instance."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> MockOutput:
        return MockOutput(value=42)


class ReturnPSRStage(Stage):
    """compute() returns a ProgramStageResult directly."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> ProgramStageResult:
        return ProgramStageResult.success(output=MockOutput(value=99))


class ReturnNoneVoidStage(Stage):
    """compute() returns None with VoidOutput — legal."""

    InputsModel = VoidInput
    OutputModel = VoidOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> None:
        return None


class ReturnNoneNonVoidStage(Stage):
    """compute() returns None with non-void OutputModel — illegal."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> None:
        return None


class ReturnWrongTypeStage(Stage):
    """compute() returns a wrong type (str instead of StageIO)."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> str:
        return "not a StageIO"


class RaiseStage(Stage):
    """compute() raises RuntimeError."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> MockOutput:
        raise RuntimeError("boom")


class SlowComputeStage(Stage):
    """compute() sleeps for a long time (for timeout tests)."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> MockOutput:
        await asyncio.sleep(10)
        return MockOutput(value=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog() -> Program:
    return Program(code="def solve(): return 42", state=ProgramState.RUNNING)


# ---------------------------------------------------------------------------
# TestStageExecuteReturnPaths
# ---------------------------------------------------------------------------


class TestStageExecuteReturnPaths:
    async def test_output_model_wrapped_in_psr(self):
        """ReturnOutputStage: result is ProgramStageResult with COMPLETED and output.value == 42."""
        stage = ReturnOutputStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert isinstance(result, ProgramStageResult)
        assert result.status == StageState.COMPLETED
        assert result.output.value == 42

    async def test_psr_passthrough(self):
        """ReturnPSRStage: returned ProgramStageResult passes through."""
        stage = ReturnPSRStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert isinstance(result, ProgramStageResult)
        assert result.status == StageState.COMPLETED
        assert result.output.value == 99

    async def test_void_none_accepted(self):
        """ReturnNoneVoidStage: None return with VoidOutput is COMPLETED."""
        stage = ReturnNoneVoidStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED

    async def test_none_non_void_caught_as_failure(self):
        """ReturnNoneNonVoidStage: None return with non-void OutputModel → FAILED."""
        stage = ReturnNoneNonVoidStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert "TypeError" in result.error.type

    async def test_wrong_type_caught_as_failure(self):
        """ReturnWrongTypeStage: wrong return type → FAILED with TypeError."""
        stage = ReturnWrongTypeStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert "TypeError" in result.error.type

    async def test_exception_caught_as_failure(self):
        """RaiseStage: RuntimeError → FAILED with StageError containing 'boom'."""
        stage = RaiseStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert "boom" in result.error.message


# ---------------------------------------------------------------------------
# TestStageExecuteTimeout
# ---------------------------------------------------------------------------


class TestStageExecuteTimeout:
    async def test_timeout_caught_as_failure(self):
        """SlowComputeStage with tiny timeout → FAILED."""
        stage = SlowComputeStage(timeout=0.01)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        # asyncio.TimeoutError is wrapped
        assert result.error is not None


# ---------------------------------------------------------------------------
# TestStageExecuteTimestamps
# ---------------------------------------------------------------------------


class TestStageExecuteTimestamps:
    async def test_started_at_and_finished_at_set(self):
        """Any stage result has started_at and finished_at timestamps."""
        stage = ReturnOutputStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.finished_at >= result.started_at

    async def test_failure_has_timestamps(self):
        """Failed stages also have timestamps."""
        stage = RaiseStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.started_at is not None
        assert result.finished_at is not None


# ---------------------------------------------------------------------------
# TestStageExecuteCleanup
# ---------------------------------------------------------------------------


class TestStageExecuteCleanup:
    async def test_inputs_cleared_after_success(self):
        """After successful execute, stage._raw_inputs is empty."""
        stage = ReturnOutputStage(timeout=5.0)
        stage.attach_inputs({})
        await stage.execute(_prog())

        assert stage._raw_inputs == {}
        assert stage._params_obj is None

    async def test_inputs_cleared_after_failure(self):
        """After failed execute, stage._raw_inputs is still cleared."""
        stage = RaiseStage(timeout=5.0)
        stage.attach_inputs({})
        await stage.execute(_prog())

        assert stage._raw_inputs == {}
        assert stage._params_obj is None

    async def test_current_inputs_hash_cleared(self):
        """After execute, stage._current_inputs_hash is None."""
        stage = ReturnOutputStage(timeout=5.0)
        stage.attach_inputs({})
        await stage.execute(_prog())

        assert stage._current_inputs_hash is None


# ---------------------------------------------------------------------------
# TestStageExecuteCache
# ---------------------------------------------------------------------------


class TestStageExecuteCache:
    async def test_input_hash_set_on_result(self):
        """For InputHashCache (default), result.input_hash is populated."""

        # Use a stage with the DEFAULT_CACHE (InputHashCache)
        class DefaultCacheStage(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput

            async def compute(self, program: Program) -> MockOutput:
                return MockOutput(value=1)

        stage = DefaultCacheStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.input_hash is not None
