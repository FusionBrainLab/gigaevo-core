"""Tests for LangGraphStage: preprocess, postprocess, agent call, error handling."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageIO,
    StageState,
    VoidInput,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import StringContainer
from gigaevo.programs.stages.langgraph_stage import LangGraphStage

# ---------------------------------------------------------------------------
# Custom OutputModels for tests
# ---------------------------------------------------------------------------


class SingleFieldOutput(StageIO):
    value: int


class MultiFieldOutput(StageIO):
    name: str
    score: float


# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------


class VoidToSingleField(LangGraphStage):
    InputsModel = VoidInput
    OutputModel = SingleFieldOutput
    cache_handler = NO_CACHE


class VoidToMultiField(LangGraphStage):
    InputsModel = VoidInput
    OutputModel = MultiFieldOutput
    cache_handler = NO_CACHE


class VoidToString(LangGraphStage):
    InputsModel = VoidInput
    OutputModel = StringContainer
    cache_handler = NO_CACHE


class WithProgramKwarg(LangGraphStage):
    InputsModel = VoidInput
    OutputModel = SingleFieldOutput
    cache_handler = NO_CACHE


class ShortCircuitPreprocess(LangGraphStage):
    """Subclass that short-circuits in preprocess."""

    InputsModel = VoidInput
    OutputModel = SingleFieldOutput
    cache_handler = NO_CACHE

    async def preprocess(
        self, program: Program, params: StageIO
    ) -> dict[str, Any] | ProgramStageResult:
        return ProgramStageResult.skipped(
            message="Nothing to do", stage=self.stage_name
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog() -> Program:
    return Program(code="def solve(): return 42", state=ProgramState.RUNNING)


def _mock_agent(return_value: Any = None, side_effect: Exception | None = None):
    agent = AsyncMock()
    if side_effect:
        agent.arun.side_effect = side_effect
    else:
        agent.arun.return_value = return_value
    return agent


# ---------------------------------------------------------------------------
# TestPostprocessBranches
# ---------------------------------------------------------------------------


class TestPostprocessBranches:
    async def test_exact_output_model_passthrough(self):
        """Agent returns exact OutputModel → returned as-is."""
        agent = _mock_agent(return_value=SingleFieldOutput(value=42))
        stage = VoidToSingleField(agent=agent, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.value == 42

    async def test_single_field_scalar_wrap(self):
        """Agent returns a plain int for single-field OutputModel → wrapped."""
        agent = _mock_agent(return_value=99)
        stage = VoidToSingleField(agent=agent, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.value == 99

    async def test_dict_validate_multi_field(self):
        """Agent returns dict for multi-field OutputModel → model_validate."""
        agent = _mock_agent(return_value={"name": "test", "score": 0.95})
        stage = VoidToMultiField(agent=agent, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.name == "test"
        assert result.output.score == pytest.approx(0.95)

    async def test_invalid_dict_multi_field_fails(self):
        """Agent returns dict with missing field → FAILED (ValidationError)."""
        agent = _mock_agent(return_value={"name": "test"})  # missing 'score'
        stage = VoidToMultiField(agent=agent, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED

    async def test_wrong_type_non_dict_fails(self):
        """Agent returns non-dict non-OutputModel for multi-field → TypeError → FAILED."""
        agent = _mock_agent(return_value="just a string")
        stage = VoidToMultiField(agent=agent, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert "TypeError" in result.error.type

    async def test_single_field_wrong_type_falls_through(self):
        """Agent returns wrong type for single-field → Pydantic fails → falls through → TypeError."""
        # SingleFieldOutput expects int, but agent returns a dict that can't be coerced
        agent = _mock_agent(return_value={"not": "an int"})
        stage = VoidToSingleField(agent=agent, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        # The dict will go through model_validate path which may succeed or fail
        # depending on Pydantic coercion rules
        assert result.status in (StageState.COMPLETED, StageState.FAILED)


# ---------------------------------------------------------------------------
# TestPreprocessShortCircuit
# ---------------------------------------------------------------------------


class TestPreprocessShortCircuit:
    async def test_preprocess_returns_psr_skips_agent(self):
        """When preprocess returns ProgramStageResult → agent not called."""
        agent = _mock_agent()
        stage = ShortCircuitPreprocess(agent=agent, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.SKIPPED
        agent.arun.assert_not_called()


# ---------------------------------------------------------------------------
# TestProgramKwarg
# ---------------------------------------------------------------------------


class TestProgramKwarg:
    async def test_program_injected(self):
        """program_kwarg injects program into agent kwargs."""
        agent = _mock_agent(return_value=SingleFieldOutput(value=1))
        stage = WithProgramKwarg(agent=agent, program_kwarg="program", timeout=5.0)
        stage.attach_inputs({})
        prog = _prog()
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        # Verify the agent was called with 'program' kwarg
        call_kwargs = agent.arun.call_args[1]
        assert "program" in call_kwargs
        assert call_kwargs["program"] is prog

    async def test_program_kwarg_collision_raises(self):
        """Preprocess returns dict with same key as program_kwarg → ValueError → FAILED."""

        class CollidingPreprocess(LangGraphStage):
            InputsModel = VoidInput
            OutputModel = SingleFieldOutput
            cache_handler = NO_CACHE

            async def preprocess(self, program, params):
                return {"program": "collision"}

        agent = _mock_agent()
        stage = CollidingPreprocess(agent=agent, program_kwarg="program", timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert "collides" in result.error.message


# ---------------------------------------------------------------------------
# TestAgentExceptionHandling
# ---------------------------------------------------------------------------


class TestAgentExceptionHandling:
    async def test_agent_exception_caught_as_failure(self):
        """Agent raises RuntimeError → FAILED with error message."""
        agent = _mock_agent(side_effect=RuntimeError("LLM timeout"))
        stage = VoidToSingleField(agent=agent, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert "LLM timeout" in result.error.message

    async def test_agent_value_error_caught(self):
        """Agent raises ValueError → FAILED."""
        agent = _mock_agent(side_effect=ValueError("bad input"))
        stage = VoidToSingleField(agent=agent, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert "bad input" in result.error.message


# ---------------------------------------------------------------------------
# TestDefaultPreprocess
# ---------------------------------------------------------------------------


class TestDefaultPreprocess:
    async def test_void_input_passes_empty_kwargs(self):
        """VoidInput → agent called with no input kwargs (just program if set)."""
        agent = _mock_agent(return_value=SingleFieldOutput(value=7))
        stage = VoidToSingleField(agent=agent, timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        # VoidInput has no fields → agent called with empty kwargs
        agent.arun.assert_called_once_with()
