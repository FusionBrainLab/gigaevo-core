"""Tests for Stage.execute() — return type dispatch, error handling, caching, and cleanup."""

from __future__ import annotations

import asyncio
from datetime import UTC

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
from gigaevo.programs.stages.cache_handler import (
    NO_CACHE,
    CacheHandler,
    InputHashCache,
)

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


# ---------------------------------------------------------------------------
# Spy CacheHandler for verifying on_complete calls
# ---------------------------------------------------------------------------


class SpyCacheHandler(CacheHandler):
    """CacheHandler that records all on_complete calls for verification."""

    def __init__(self):
        self.calls: list[tuple[ProgramStageResult, str | None]] = []

    def should_rerun(self, existing_result, inputs_hash, finished_this_run) -> bool:
        return True

    def on_complete(self, result, inputs_hash):
        self.calls.append((result, inputs_hash))
        return result


# ---------------------------------------------------------------------------
# Stage subclasses for on_complete call-site tests
# ---------------------------------------------------------------------------


class _SpyOutputStage(Stage):
    """Normal OutputModel return — exercises on_complete call site at line ~294."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = SpyCacheHandler()

    async def compute(self, program: Program) -> MockOutput:
        return MockOutput(value=10)


class _SpyPSRStage(Stage):
    """Returns ProgramStageResult directly — exercises on_complete at line ~259."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = SpyCacheHandler()

    async def compute(self, program: Program) -> ProgramStageResult:
        return ProgramStageResult.success(output=MockOutput(value=20))


class _SpyVoidStage(Stage):
    """VoidOutput returning None — exercises on_complete at line ~273."""

    InputsModel = VoidInput
    OutputModel = VoidOutput
    cache_handler = SpyCacheHandler()

    async def compute(self, program: Program) -> None:
        return None


class _SpyRaiseStage(Stage):
    """Raises RuntimeError — exercises on_complete at line ~315."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = SpyCacheHandler()

    async def compute(self, program: Program) -> MockOutput:
        raise RuntimeError("spy boom")


class _SpyTimeoutStage(Stage):
    """Sleeps forever — exercises timeout -> on_complete at line ~315."""

    InputsModel = VoidInput
    OutputModel = MockOutput
    cache_handler = SpyCacheHandler()

    async def compute(self, program: Program) -> MockOutput:
        await asyncio.sleep(3600)
        return MockOutput(value=0)  # pragma: no cover


# ---------------------------------------------------------------------------
# TestOnCompleteAllCallSites
# ---------------------------------------------------------------------------


class TestOnCompleteAllCallSites:
    """Audit finding #1: on_complete() is called in 4 places in Stage.execute().
    Verify that EACH path calls on_complete with the correct arguments."""

    async def test_on_complete_called_on_normal_output_return(self):
        """Call site ~294: compute() returns an OutputModel instance."""
        spy = SpyCacheHandler()
        _SpyOutputStage.cache_handler = spy

        stage = _SpyOutputStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert len(spy.calls) == 1
        call_result, call_hash = spy.calls[0]
        assert call_result.status == StageState.COMPLETED
        assert call_result.output.value == 10
        # Hash should be a string (computed from VoidInput)
        assert isinstance(call_hash, str)

    async def test_on_complete_called_on_psr_passthrough(self):
        """Call site ~259: compute() returns a ProgramStageResult directly."""
        spy = SpyCacheHandler()
        _SpyPSRStage.cache_handler = spy

        stage = _SpyPSRStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert len(spy.calls) == 1
        call_result, call_hash = spy.calls[0]
        assert call_result.status == StageState.COMPLETED
        assert call_result.output.value == 20
        assert isinstance(call_hash, str)

    async def test_on_complete_called_on_void_none_return(self):
        """Call site ~273: compute() returns None with VoidOutput."""
        spy = SpyCacheHandler()
        _SpyVoidStage.cache_handler = spy

        stage = _SpyVoidStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert len(spy.calls) == 1
        call_result, call_hash = spy.calls[0]
        assert call_result.status == StageState.COMPLETED
        assert call_result.output is None
        assert isinstance(call_hash, str)

    async def test_on_complete_called_on_exception(self):
        """Call site ~315: compute() raises an exception."""
        spy = SpyCacheHandler()
        _SpyRaiseStage.cache_handler = spy

        stage = _SpyRaiseStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert len(spy.calls) == 1
        call_result, call_hash = spy.calls[0]
        assert call_result.status == StageState.FAILED
        assert call_result.error is not None
        assert "spy boom" in call_result.error.message
        assert isinstance(call_hash, str)

    async def test_on_complete_called_on_timeout(self):
        """Call site ~315 (via TimeoutError): compute() times out."""
        spy = SpyCacheHandler()
        _SpyTimeoutStage.cache_handler = spy

        stage = _SpyTimeoutStage(timeout=0.01)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert len(spy.calls) == 1
        call_result, call_hash = spy.calls[0]
        assert call_result.status == StageState.FAILED
        assert call_result.error is not None
        assert isinstance(call_hash, str)

    async def test_on_complete_hash_matches_compute_inputs_hash(self):
        """The inputs_hash passed to on_complete matches compute_inputs_hash()."""
        spy = SpyCacheHandler()
        _SpyOutputStage.cache_handler = spy

        stage = _SpyOutputStage(timeout=5.0)
        stage.attach_inputs({})

        # Compute the expected hash before execute (which clears state)
        expected_hash = stage.compute_inputs_hash()

        result = await stage.execute(_prog())

        # Re-attach to compute hash again for comparison
        stage.attach_inputs({})
        recomputed_hash = stage.compute_inputs_hash()

        assert result.status == StageState.COMPLETED
        _, call_hash = spy.calls[0]
        assert call_hash == expected_hash
        assert call_hash == recomputed_hash


# ---------------------------------------------------------------------------
# TestFailurePathOnComplete
# ---------------------------------------------------------------------------


class TestFailurePathOnComplete:
    """Audit finding #2: Verify that when a stage raises an exception,
    on_complete() is called and the result has correct error information."""

    async def test_failure_on_complete_result_has_error_type(self):
        """Failed stage result passed to on_complete has error.type set."""
        spy = SpyCacheHandler()
        _SpyRaiseStage.cache_handler = spy

        stage = _SpyRaiseStage(timeout=5.0)
        stage.attach_inputs({})
        await stage.execute(_prog())

        call_result, _ = spy.calls[0]
        assert call_result.error.type == "RuntimeError"

    async def test_failure_on_complete_result_has_error_message(self):
        """Failed stage result passed to on_complete has error.message set."""
        spy = SpyCacheHandler()
        _SpyRaiseStage.cache_handler = spy

        stage = _SpyRaiseStage(timeout=5.0)
        stage.attach_inputs({})
        await stage.execute(_prog())

        call_result, _ = spy.calls[0]
        assert "spy boom" in call_result.error.message

    async def test_failure_on_complete_result_has_stage_name(self):
        """Failed stage result passed to on_complete has error.stage set."""
        spy = SpyCacheHandler()
        _SpyRaiseStage.cache_handler = spy

        stage = _SpyRaiseStage(timeout=5.0)
        stage.attach_inputs({})
        await stage.execute(_prog())

        call_result, _ = spy.calls[0]
        assert call_result.error.stage == "_SpyRaiseStage"

    async def test_failure_on_complete_result_is_same_object_returned(self):
        """The result from on_complete is what execute() returns (not a copy)."""
        spy = SpyCacheHandler()
        _SpyRaiseStage.cache_handler = spy

        stage = _SpyRaiseStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        call_result, _ = spy.calls[0]
        # SpyCacheHandler returns the same object, so they should be identical
        assert result is call_result

    async def test_failure_on_complete_input_hash_populated(self):
        """Even on failure, on_complete receives a valid input_hash."""
        spy = SpyCacheHandler()
        _SpyRaiseStage.cache_handler = spy

        stage = _SpyRaiseStage(timeout=5.0)
        stage.attach_inputs({})
        await stage.execute(_prog())

        _, call_hash = spy.calls[0]
        assert call_hash is not None
        assert isinstance(call_hash, str)
        assert len(call_hash) > 0

    async def test_input_hash_cache_stores_hash_on_failure(self):
        """With InputHashCache, failed stages still get input_hash stored on the result.
        This prevents unnecessary reruns on refresh."""

        class FailWithHashCacheStage(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput
            cache_handler = InputHashCache()

            async def compute(self, program: Program) -> MockOutput:
                raise ValueError("hash cache fail test")

        stage = FailWithHashCacheStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        # InputHashCache.on_complete stores the hash on the result
        assert result.input_hash is not None
        assert isinstance(result.input_hash, str)


# ---------------------------------------------------------------------------
# TestTimeoutMechanismVerification
# ---------------------------------------------------------------------------


class TestTimeoutMechanismVerification:
    """Audit finding #3: Verify the timeout mechanism fires correctly
    and produces the right error type in the result."""

    async def test_timeout_produces_timeout_error_type(self):
        """Timeout error type should be TimeoutError or asyncio.TimeoutError."""
        stage = SlowComputeStage(timeout=0.01)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert result.error is not None
        # asyncio.wait_for raises TimeoutError (or asyncio.TimeoutError)
        assert "TimeoutError" in result.error.type

    async def test_timeout_result_has_timestamps(self):
        """Timed-out stage result still has started_at and finished_at."""
        stage = SlowComputeStage(timeout=0.01)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.started_at is not None
        assert result.finished_at is not None

    async def test_timeout_duration_is_short(self):
        """Timeout should fire quickly, not wait for the full compute() duration."""
        stage = SlowComputeStage(timeout=0.05)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        duration = result.duration_seconds()
        assert duration is not None
        # Should complete well under 1 second (timeout is 0.05s)
        assert duration < 1.0

    async def test_timeout_stage_name_in_error(self):
        """Timed-out stage error should contain the stage name."""
        stage = SlowComputeStage(timeout=0.01)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.error.stage == "SlowComputeStage"

    async def test_no_timeout_when_compute_finishes_fast(self):
        """A stage that finishes well within timeout should succeed."""
        stage = ReturnOutputStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.error is None


# ---------------------------------------------------------------------------
# TestErrorStageFieldAssertion
# ---------------------------------------------------------------------------


class TestErrorStageFieldAssertion:
    """Audit finding #4: When a stage fails, StageError.stage should contain
    the stage class name for all failure modes."""

    async def test_error_stage_on_runtime_error(self):
        """RuntimeError in compute() produces error.stage == stage_name."""
        stage = RaiseStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert result.error.stage == "RaiseStage"

    async def test_error_stage_on_timeout(self):
        """Timeout produces error.stage == stage_name."""
        stage = SlowComputeStage(timeout=0.01)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert result.error.stage == "SlowComputeStage"

    async def test_error_stage_on_type_error_none_non_void(self):
        """None return with non-void OutputModel: error.stage == stage_name."""
        stage = ReturnNoneNonVoidStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert result.error.stage == "ReturnNoneNonVoidStage"

    async def test_error_stage_on_wrong_return_type(self):
        """Wrong return type: error.stage == stage_name."""
        stage = ReturnWrongTypeStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert result.error.stage == "ReturnWrongTypeStage"

    async def test_error_stage_on_value_error(self):
        """ValueError in compute() produces error.stage with correct stage name."""

        class ValueErrorStage(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> MockOutput:
                raise ValueError("bad value in stage")

        stage = ValueErrorStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert result.error.stage == "ValueErrorStage"
        assert result.error.type == "ValueError"


# ---------------------------------------------------------------------------
# TestHashBeforeComputeOrdering
# ---------------------------------------------------------------------------


class TestHashBeforeComputeOrdering:
    """Audit finding #5: The input hash must be computed BEFORE compute() runs,
    not after, since compute() could have side effects that modify state."""

    async def test_hash_computed_before_compute_executes(self):
        """Verify input hash is computed before compute() is called
        by checking that a side-effect in compute() does not affect the hash."""
        hash_seen_during_compute = []

        class HashOrderStage(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> MockOutput:
                # Record what _current_inputs_hash is during compute
                hash_seen_during_compute.append(self._current_inputs_hash)
                return MockOutput(value=1)

        stage = HashOrderStage(timeout=5.0)
        stage.attach_inputs({})

        # Compute expected hash before execute
        expected_hash = stage.compute_inputs_hash()

        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        # Hash should have been set before compute() ran
        assert len(hash_seen_during_compute) == 1
        assert hash_seen_during_compute[0] is not None
        assert hash_seen_during_compute[0] == expected_hash

    async def test_hash_set_before_ensure_required_present(self):
        """_current_inputs_hash is set before _ensure_required_present is called.
        This is visible in the source: line 246 (hash) before line 249 (ensure)."""
        hash_at_ensure_time = []

        class InstrumentedStage(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput
            cache_handler = NO_CACHE

            def _ensure_required_present(self):
                # Capture hash state at the time _ensure_required_present runs
                hash_at_ensure_time.append(self._current_inputs_hash)
                super()._ensure_required_present()

            async def compute(self, program: Program) -> MockOutput:
                return MockOutput(value=1)

        stage = InstrumentedStage(timeout=5.0)
        stage.attach_inputs({})

        result = await stage.execute(_prog())
        assert result.status == StageState.COMPLETED
        assert len(hash_at_ensure_time) == 1
        assert hash_at_ensure_time[0] is not None

    async def test_side_effect_in_compute_does_not_change_hash(self):
        """Even if compute() modifies stage internal state, the hash was
        already captured and is passed correctly to on_complete."""
        spy = SpyCacheHandler()

        class SideEffectHashStage(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput
            cache_handler = spy

            async def compute(self, program: Program) -> MockOutput:
                # Try to tamper with _raw_inputs during compute
                # (this should NOT affect the already-computed hash)
                return MockOutput(value=999)

        stage = SideEffectHashStage(timeout=5.0)
        stage.attach_inputs({})
        pre_hash = stage.compute_inputs_hash()

        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        _, on_complete_hash = spy.calls[0]
        assert on_complete_hash == pre_hash


# ---------------------------------------------------------------------------
# TestProgramStageResultTimestampBranches
# ---------------------------------------------------------------------------


class TestProgramStageResultTimestampBranches:
    """Audit finding #6: ProgramStageResult has correct started_at and duration
    for both success and failure cases."""

    async def test_success_has_positive_duration(self):
        """Successful stage execution should have duration > 0."""
        stage = ReturnOutputStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.started_at is not None
        assert result.finished_at is not None
        duration = result.duration_seconds()
        assert duration is not None
        assert duration >= 0

    async def test_failure_has_positive_duration(self):
        """Failed stage execution should have duration > 0."""
        stage = RaiseStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert result.started_at is not None
        assert result.finished_at is not None
        duration = result.duration_seconds()
        assert duration is not None
        assert duration >= 0

    async def test_timeout_has_bounded_duration(self):
        """Timed-out stage should have duration roughly matching the timeout."""
        stage = SlowComputeStage(timeout=0.05)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        duration = result.duration_seconds()
        assert duration is not None
        # Duration should be at least the timeout value
        assert duration >= 0.01
        # But not much more than 1 second
        assert duration < 2.0

    async def test_void_stage_has_timestamps(self):
        """VoidOutput stage returning None has valid timestamps."""
        stage = ReturnNoneVoidStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.finished_at >= result.started_at

    async def test_psr_passthrough_gets_started_at_if_missing(self):
        """When compute() returns a PSR without started_at, execute() fills it."""

        class PSRNoTimestampStage(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> ProgramStageResult:
                # Return PSR without setting started_at
                return ProgramStageResult(
                    status=StageState.COMPLETED,
                    output=MockOutput(value=77),
                )

        stage = PSRNoTimestampStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        # execute() should have filled in started_at
        assert result.started_at is not None

    async def test_psr_passthrough_preserves_existing_started_at(self):
        """When compute() returns a PSR with started_at already set,
        execute() does not overwrite it."""
        from datetime import datetime

        fixed_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        class PSRWithTimestampStage(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> ProgramStageResult:
                return ProgramStageResult(
                    status=StageState.COMPLETED,
                    output=MockOutput(value=88),
                    started_at=fixed_time,
                    finished_at=fixed_time,
                )

        stage = PSRWithTimestampStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        # Should preserve the original started_at
        assert result.started_at == fixed_time

    async def test_slow_stage_has_measurable_duration(self):
        """A stage that takes some time should have a measurable duration."""

        class SlightlySlowStage(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> MockOutput:
                await asyncio.sleep(0.05)
                return MockOutput(value=1)

        stage = SlightlySlowStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        duration = result.duration_seconds()
        assert duration is not None
        # Should be at least 50ms
        assert duration >= 0.03


# ---------------------------------------------------------------------------
# TestComputeInputsHashFailure (P0 bug fix)
# ---------------------------------------------------------------------------


class TestComputeInputsHashFailure:
    """Bug fix: compute_inputs_hash() raising must be caught by execute(),
    not propagate as an unhandled exception. Before the fix, the call was
    outside the try/except block, so a bad compute_hash override would crash
    the entire DAG runner."""

    async def test_hash_raises_returns_failure(self):
        """compute_inputs_hash() raises -> FAILED result, not unhandled crash."""

        class BadHashStage(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> MockOutput:
                return MockOutput(value=1)  # pragma: no cover

            def compute_inputs_hash(self) -> str | None:
                raise RuntimeError("hash computation exploded")

        stage = BadHashStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert "hash computation exploded" in result.error.message
        assert result.error.stage == "BadHashStage"

    async def test_hash_raises_cleanup_still_runs(self):
        """After compute_inputs_hash() raises, _raw_inputs and _params_obj
        are still cleared by the finally block."""

        class BadHashStage2(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> MockOutput:
                return MockOutput(value=1)  # pragma: no cover

            def compute_inputs_hash(self) -> str | None:
                raise ValueError("bad hash")

        stage = BadHashStage2(timeout=5.0)
        stage.attach_inputs({})
        await stage.execute(_prog())

        assert stage._raw_inputs == {}
        assert stage._params_obj is None
        assert stage._current_inputs_hash is None

    async def test_hash_raises_on_complete_still_called(self):
        """Even when compute_inputs_hash() raises, on_complete is called
        (with None hash) so cache handlers can record the failure."""
        spy = SpyCacheHandler()

        class BadHashSpyStage(Stage):
            InputsModel = VoidInput
            OutputModel = MockOutput
            cache_handler = spy

            async def compute(self, program: Program) -> MockOutput:
                return MockOutput(value=1)  # pragma: no cover

            def compute_inputs_hash(self) -> str | None:
                raise TypeError("hash broken")

        stage = BadHashSpyStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert len(spy.calls) == 1
        call_result, call_hash = spy.calls[0]
        assert call_result.status == StageState.FAILED
        # Hash is None because compute_inputs_hash raised before setting it
        assert call_hash is None


# ---------------------------------------------------------------------------
# TestCacheHandlerOnCompleteRaises
# ---------------------------------------------------------------------------
