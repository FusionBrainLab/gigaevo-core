"""Tests for Stage.partial_result() — graceful-shutdown salvage hook.

When compute() is cancelled (asyncio.TimeoutError from wait_for, or external
CancelledError), Stage.execute() calls self.partial_result(program). If the
override returns a non-None OutputModel instance, the stage result is
COMPLETED with that output instead of FAILED. If partial_result returns None
(the default), the stage fails as before.
"""

from __future__ import annotations

import asyncio

from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageIO,
    StageState,
    VoidInput,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE


class _CounterOutput(StageIO):
    counter: int = 0


def _prog() -> Program:
    return Program(code="def solve(): return 0", state=ProgramState.RUNNING)


class _CountingSlowStage(Stage):
    """Increments self._counter every 10 ms, never returns on its own."""

    InputsModel = VoidInput
    OutputModel = _CounterOutput
    cache_handler = NO_CACHE

    def __init__(self, *, timeout: float):
        super().__init__(timeout=timeout)
        self._counter = 0

    async def compute(self, program: Program) -> _CounterOutput:
        while True:
            self._counter += 1
            await asyncio.sleep(0.01)

    async def partial_result(self, program: Program) -> _CounterOutput | None:
        if self._counter == 0:
            return None
        return _CounterOutput(counter=self._counter)


class _NoSalvageSlowStage(Stage):
    """Same compute() loop but inherits the default partial_result (None)."""

    InputsModel = VoidInput
    OutputModel = _CounterOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> _CounterOutput:
        while True:
            await asyncio.sleep(0.01)


class _SalvageReturnsWrongTypeStage(Stage):
    """partial_result returns a value that is not an OutputModel — must fail."""

    InputsModel = VoidInput
    OutputModel = _CounterOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> _CounterOutput:
        await asyncio.sleep(10)
        return _CounterOutput()

    async def partial_result(self, program: Program):  # noqa: ANN201 — intentional bad return
        return "not a StageIO"


class _SalvageRaisesStage(Stage):
    """partial_result itself raises — must fall through to FAILED, not crash."""

    InputsModel = VoidInput
    OutputModel = _CounterOutput
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> _CounterOutput:
        await asyncio.sleep(10)
        return _CounterOutput()

    async def partial_result(self, program: Program) -> _CounterOutput | None:
        raise RuntimeError("salvage exploded")


class TestStagePartialResultOnTimeout:
    async def test_salvage_returns_completed_with_partial_output(self):
        """Slow stage with timeout → partial_result fires → COMPLETED with counter > 0."""
        stage = _CountingSlowStage(timeout=0.2)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert isinstance(result, ProgramStageResult)
        assert result.status == StageState.COMPLETED
        assert result.output is not None
        assert result.output.counter > 0

    async def test_default_partial_result_preserves_failed(self):
        """Default partial_result returns None → behaviour unchanged → FAILED."""
        stage = _NoSalvageSlowStage(timeout=0.05)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert result.error is not None
        assert "TimeoutError" in result.error.type

    async def test_salvage_wrong_type_falls_through_to_failed(self):
        """partial_result returning non-OutputModel must NOT be silently accepted."""
        stage = _SalvageReturnsWrongTypeStage(timeout=0.05)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED

    async def test_salvage_exception_falls_through_to_failed(self):
        """Exception from partial_result must not crash execute() — fail as before."""
        stage = _SalvageRaisesStage(timeout=0.05)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED


class TestStagePartialResultOnCancellation:
    async def test_external_cancel_triggers_salvage(self):
        """External CancelledError (e.g. DAG cancellation) also triggers salvage.

        Schedule execute() as a task, let compute() spin briefly, then cancel
        the task. The salvage hook should still fire on the inner CancelledError
        and return COMPLETED with whatever counter accumulated.
        """
        stage = _CountingSlowStage(timeout=60.0)  # plenty of headroom
        stage.attach_inputs({})

        task = asyncio.create_task(stage.execute(_prog()))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            result = await task
        except asyncio.CancelledError:
            result = None

        assert result is not None, (
            "execute() must catch its own CancelledError and salvage, "
            "not propagate the cancellation out"
        )
        assert result.status == StageState.COMPLETED
        assert result.output.counter > 0
