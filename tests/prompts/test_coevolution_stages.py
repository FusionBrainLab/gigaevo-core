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

    @pytest.mark.asyncio
    async def test_str_entrypoint_has_no_user_text(
        self, simple_prompt_program: Program
    ):
        """str-returning entrypoint sets user_text=None."""
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})
        result = await stage.compute(simple_prompt_program)
        assert result.user_text is None

    @pytest.mark.asyncio
    async def test_dict_entrypoint_system_and_user(self):
        """dict entrypoint with system+user keys extracts both texts."""
        code = """
def entrypoint() -> dict:
    return {
        "system": "You are a mutation expert for {task_description}.",
        "user": "Mutate {count} programs.\\n\\n{parent_blocks}",
    }
"""
        program = Program(code=code)
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})
        result = await stage.compute(program)
        assert result.prompt_text == "You are a mutation expert for {task_description}."
        assert result.user_text == "Mutate {count} programs.\n\n{parent_blocks}"
        assert len(result.prompt_id) == 16

    @pytest.mark.asyncio
    async def test_dict_entrypoint_system_only(self):
        """dict entrypoint with only 'system' key sets user_text=None."""
        code = """
def entrypoint() -> dict:
    return {"system": "System prompt here."}
"""
        program = Program(code=code)
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})
        result = await stage.compute(program)
        assert result.prompt_text == "System prompt here."
        assert result.user_text is None

    @pytest.mark.asyncio
    async def test_dict_entrypoint_missing_system_key(self):
        """dict entrypoint without 'system' key raises ValueError."""
        code = """
def entrypoint() -> dict:
    return {"user": "Some user prompt."}
"""
        program = Program(code=code)
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})
        with pytest.raises(ValueError, match="non-empty 'system' key"):
            await stage.compute(program)

    @pytest.mark.asyncio
    async def test_dict_entrypoint_empty_user_raises(self):
        """dict entrypoint with empty 'user' value raises ValueError."""
        code = """
def entrypoint() -> dict:
    return {"system": "Valid system.", "user": "   "}
"""
        program = Program(code=code)
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})
        with pytest.raises(ValueError, match="non-empty str"):
            await stage.compute(program)

    @pytest.mark.asyncio
    async def test_dict_prompt_id_based_on_system_text(self):
        """prompt_id is hash of system text, not user text."""
        code_a = """
def entrypoint() -> dict:
    return {"system": "Same system prompt.", "user": "User A."}
"""
        code_b = """
def entrypoint() -> dict:
    return {"system": "Same system prompt.", "user": "User B."}
"""
        stage = PromptExecutionStage(timeout=30.0)
        stage.attach_inputs({})
        result_a = await stage.compute(Program(code=code_a))
        result_b = await stage.compute(Program(code=code_b))
        assert result_a.prompt_id == result_b.prompt_id  # Same system → same ID


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
        """compute() returns optimistic default fitness when no stats available."""
        mock_stats_provider.get_stats = AsyncMock(
            return_value=PromptMutationStats(trials=0, successes=0, success_rate=0.0)
        )

        stage = PromptFitnessStage(stats_provider=mock_stats_provider, timeout=30.0)

        program = Program(code='def entrypoint() -> str: return "test"')
        execution_output = PromptExecutionOutput(prompt_text="test", prompt_id="xyz789")
        stage.attach_inputs({"execution_output": execution_output})

        result = await stage.compute(program)

        assert result.data["fitness"] == 0.01  # optimistic default
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
        assert output.user_text is None  # optional, defaults to None

    def test_creation_with_user_text(self):
        """PromptExecutionOutput can include user_text."""
        output = PromptExecutionOutput(
            prompt_text="System prompt", user_text="User {count}", prompt_id="def"
        )
        assert output.prompt_text == "System prompt"
        assert output.user_text == "User {count}"
        assert output.prompt_id == "def"
