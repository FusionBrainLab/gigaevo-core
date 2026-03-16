"""Tests for prompt coevolution stages (PromptExecutionStage, PromptFitnessStage)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.programs.program import Program
from gigaevo.prompts.coevolution.stages import (
    PromptExecutionOutput,
    PromptExecutionStage,
    PromptFitnessStage,
)
from gigaevo.prompts.coevolution.stats import PromptMutationStats, PromptStatsProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_prompt_program() -> Program:
    """Program that returns a valid prompt string."""
    code = """
def entrypoint() -> str:
    return "You are a mutation expert. Optimize the given program."
"""
    return Program(code=code)


@pytest.fixture
def broken_prompt_program() -> Program:
    """Program with no entrypoint() function."""
    code = "x = 1"
    return Program(code=code)


@pytest.fixture
def invalid_return_program() -> Program:
    """Program where entrypoint() returns non-string."""
    code = """
def entrypoint() -> int:
    return 42
"""
    return Program(code=code)


@pytest.fixture
def empty_prompt_program() -> Program:
    """Program where entrypoint() returns empty string."""
    code = """
def entrypoint() -> str:
    return ""
"""
    return Program(code=code)


@pytest.fixture
def mock_stats_provider() -> MagicMock:
    """Mock PromptStatsProvider."""
    return MagicMock(spec=PromptStatsProvider)


# ---------------------------------------------------------------------------
# PromptExecutionStage Tests
# ---------------------------------------------------------------------------


class TestPromptExecutionStage:
    """Tests for PromptExecutionStage."""

    def test_initialization(self):
        """PromptExecutionStage can be initialized."""
        stage = PromptExecutionStage(timeout=30.0)
        assert stage is not None

    @pytest.mark.asyncio
    async def test_execute_valid_program(self, simple_prompt_program: Program):
        """compute() executes entrypoint() and stores prompt text."""
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})  # No inputs needed
        result = await stage.compute(simple_prompt_program)
        assert (
            result.prompt_text
            == "You are a mutation expert. Optimize the given program."
        )
        assert len(result.prompt_id) == 16  # SHA256[:16]

    @pytest.mark.asyncio
    async def test_execute_no_entrypoint(self, broken_prompt_program: Program):
        """compute() raises when program has no entrypoint()."""
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})
        with pytest.raises(ValueError, match="no callable entrypoint"):
            await stage.compute(broken_prompt_program)

    @pytest.mark.asyncio
    async def test_execute_non_string_return(self, invalid_return_program: Program):
        """compute() raises when entrypoint() returns non-string."""
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})
        with pytest.raises(ValueError, match="must return str"):
            await stage.compute(invalid_return_program)

    @pytest.mark.asyncio
    async def test_execute_empty_string(self, empty_prompt_program: Program):
        """compute() raises when entrypoint() returns empty string."""
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})
        with pytest.raises(ValueError, match="returned empty string"):
            await stage.compute(empty_prompt_program)

    @pytest.mark.asyncio
    async def test_execute_syntax_error(self):
        """compute() raises when program has syntax error."""
        program = Program(code="def entrypoint() -> str\n    return x")  # Missing :
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})
        with pytest.raises(ValueError, match="syntax error"):
            await stage.compute(program)

    @pytest.mark.asyncio
    async def test_execute_runtime_error(self):
        """compute() raises when entrypoint() raises exception."""
        code = """
def entrypoint() -> str:
    raise RuntimeError("Intentional error")
"""
        program = Program(code=code)
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})
        with pytest.raises(ValueError, match="raised an exception"):
            await stage.compute(program)

    @pytest.mark.asyncio
    async def test_prompt_id_stability(self, simple_prompt_program: Program):
        """compute() produces stable prompt IDs for same text."""
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})
        result1 = await stage.compute(simple_prompt_program)
        result2 = await stage.compute(simple_prompt_program)
        assert result1.prompt_id == result2.prompt_id


# ---------------------------------------------------------------------------
# PromptFitnessStage Tests
# ---------------------------------------------------------------------------


class TestPromptFitnessStage:
    """Tests for PromptFitnessStage."""

    def test_initialization(self, mock_stats_provider: MagicMock):
        """PromptFitnessStage can be initialized."""
        stage = PromptFitnessStage(stats_provider=mock_stats_provider, timeout=30.0)
        assert stage is not None

    @pytest.mark.asyncio
    async def test_compute_with_stats(self, mock_stats_provider: MagicMock):
        """compute() reads stats and returns metrics."""
        # Mock stats provider
        mock_stats_provider.get_stats = AsyncMock(
            return_value=PromptMutationStats(trials=10, successes=7, success_rate=0.7)
        )

        stage = PromptFitnessStage(stats_provider=mock_stats_provider, timeout=30.0)

        # Create program with execution output
        program = Program(code='def entrypoint() -> str: return "test prompt"')
        execution_output = PromptExecutionOutput(
            prompt_text="test prompt", prompt_id="abc123"
        )

        # Create inputs
        class MockInputs:
            execution_output = execution_output

        stage.attach_inputs({"execution_output": execution_output})

        # Execute
        result = await stage.compute(program)

        # Verify
        assert result.data["fitness"] == 0.7
        assert result.data["is_valid"] == 1.0
        assert result.data["prompt_length"] == len("test prompt")

        # Verify stats were fetched with prompt_id
        mock_stats_provider.get_stats.assert_called_once_with("abc123")

    @pytest.mark.asyncio
    async def test_compute_no_stats(self, mock_stats_provider: MagicMock):
        """compute() returns 0.0 fitness when no stats available."""
        mock_stats_provider.get_stats = AsyncMock(
            return_value=PromptMutationStats(trials=0, successes=0, success_rate=0.0)
        )

        stage = PromptFitnessStage(stats_provider=mock_stats_provider, timeout=30.0)

        program = Program(code='def entrypoint() -> str: return "test"')
        execution_output = PromptExecutionOutput(prompt_text="test", prompt_id="xyz789")
        stage.attach_inputs({"execution_output": execution_output})

        result = await stage.compute(program)

        assert result.data["fitness"] == 0.0
        assert result.data["is_valid"] == 1.0

    @pytest.mark.asyncio
    async def test_compute_stores_metrics(self, mock_stats_provider: MagicMock):
        """compute() stores metrics on the program."""
        mock_stats_provider.get_stats = AsyncMock(
            return_value=PromptMutationStats(trials=5, successes=3, success_rate=0.6)
        )

        stage = PromptFitnessStage(stats_provider=mock_stats_provider, timeout=30.0)

        program = Program(code='def entrypoint() -> str: return "prompt"')
        execution_output = PromptExecutionOutput(prompt_text="prompt", prompt_id="pid")
        stage.attach_inputs({"execution_output": execution_output})

        await stage.compute(program)

        # Metrics should be stored on program
        assert "fitness" in program.metrics
        assert program.metrics["fitness"] == 0.6


# ---------------------------------------------------------------------------
# PromptExecutionOutput Tests
# ---------------------------------------------------------------------------


class TestPromptExecutionOutput:
    """Tests for PromptExecutionOutput."""

    def test_creation(self):
        """PromptExecutionOutput can be created."""
        output = PromptExecutionOutput(prompt_text="Hello", prompt_id="abc")
        assert output.prompt_text == "Hello"
        assert output.prompt_id == "abc"
