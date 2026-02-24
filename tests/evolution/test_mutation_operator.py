"""Tests for LLMMutationOperator with mocked LLM agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator
from gigaevo.exceptions import MutationError
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(code: str = "def solve(): return 42", **metrics) -> Program:
    p = Program(code=code, state=ProgramState.DONE)
    if metrics:
        p.add_metrics(metrics)
    return p


def _make_problem_context():
    """Build a minimal mock ProblemContext."""
    from gigaevo.programs.metrics.context import MetricsContext, MetricSpec

    ctx = MetricsContext(
        specs={
            "score": MetricSpec(
                description="test score",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=100.0,
            ),
        }
    )
    pc = MagicMock()
    pc.task_description = "Test task"
    pc.metrics_context = ctx
    return pc


def _make_operator(agent_mock, llm_mock=None, *, mode="rewrite", strip=False):
    """Build an LLMMutationOperator with a mocked agent and LLM."""
    if llm_mock is None:
        llm_mock = MagicMock()
        llm_mock.get_last_model.return_value = "test-model"
        llm_mock.on_mutation_outcome = MagicMock()

    with patch(
        "gigaevo.evolution.mutation.mutation_operator.create_mutation_agent",
        return_value=agent_mock,
    ):
        op = LLMMutationOperator(
            llm_wrapper=llm_mock,
            problem_context=_make_problem_context(),
            mutation_mode=mode,
            strip_comments_and_docstrings=strip,
        )
    return op


# ---------------------------------------------------------------------------
# TestMutateSingle
# ---------------------------------------------------------------------------


class TestMutateSingle:
    async def test_successful_mutation_returns_spec(self):
        """Agent returns valid code → MutationSpec with code and parents."""
        agent = AsyncMock()
        agent.arun.return_value = {"code": "def f(): return 1"}
        op = _make_operator(agent)

        parent = _prog()
        result = await op.mutate_single([parent])

        assert isinstance(result, MutationSpec)
        assert result.code == "def f(): return 1"
        assert result.parents == [parent]

    async def test_empty_code_raises_mutation_error(self):
        """Agent returns empty code → MutationError."""
        agent = AsyncMock()
        agent.arun.return_value = {"code": ""}
        op = _make_operator(agent)

        with pytest.raises(MutationError, match="Failed to mutate"):
            await op.mutate_single([_prog()])

    async def test_whitespace_only_code_raises(self):
        """Agent returns whitespace-only code → MutationError."""
        agent = AsyncMock()
        agent.arun.return_value = {"code": "  \n "}
        op = _make_operator(agent)

        with pytest.raises(MutationError, match="Failed to mutate"):
            await op.mutate_single([_prog()])

    async def test_no_parents_returns_none(self):
        """Empty parent list → None."""
        agent = AsyncMock()
        op = _make_operator(agent)

        result = await op.mutate_single([])

        assert result is None
        agent.arun.assert_not_called()

    async def test_diff_mode_with_multiple_parents_raises(self):
        """mode='diff', 2 parents → MutationError."""
        agent = AsyncMock()
        op = _make_operator(agent, mode="diff")

        with pytest.raises(MutationError, match="exactly 1 parent"):
            await op.mutate_single([_prog(), _prog()])

    async def test_diff_mode_with_one_parent_succeeds(self):
        """mode='diff', 1 parent → MutationSpec."""
        agent = AsyncMock()
        agent.arun.return_value = {"code": "def f(): return 2"}
        op = _make_operator(agent, mode="diff")

        result = await op.mutate_single([_prog()])

        assert isinstance(result, MutationSpec)
        assert result.code == "def f(): return 2"

    async def test_agent_exception_wrapped_in_mutation_error(self):
        """Agent.arun raises RuntimeError → wrapped in MutationError."""
        agent = AsyncMock()
        agent.arun.side_effect = RuntimeError("LLM timeout")
        op = _make_operator(agent)

        with pytest.raises(MutationError, match="LLM timeout"):
            await op.mutate_single([_prog()])

    async def test_metadata_includes_model_name(self):
        """llm_wrapper.get_last_model() → metadata has 'mutation_model'."""
        agent = AsyncMock()
        agent.arun.return_value = {"code": "def f(): return 3"}
        llm = MagicMock()
        llm.get_last_model.return_value = "gpt-4"
        llm.on_mutation_outcome = MagicMock()
        op = _make_operator(agent, llm_mock=llm)

        result = await op.mutate_single([_prog()])

        assert result.metadata.get("mutation_model") == "gpt-4"

    async def test_structured_output_captured(self):
        """Agent returns structured_output → metadata includes it."""
        agent = AsyncMock()
        agent.arun.return_value = {
            "code": "def f(): return 4",
            "structured_output": {"archetype": "local_search"},
            "archetype": "local_search",
        }
        op = _make_operator(agent)

        result = await op.mutate_single([_prog()])

        from gigaevo.llm.agents.mutation import MUTATION_OUTPUT_METADATA_KEY

        assert MUTATION_OUTPUT_METADATA_KEY in result.metadata
        assert (
            result.metadata[MUTATION_OUTPUT_METADATA_KEY]["archetype"] == "local_search"
        )


# ---------------------------------------------------------------------------
# TestCanonicalizeCode
# ---------------------------------------------------------------------------


class TestCanonicalizeCode:
    def test_removes_docstrings(self):
        """Code with docstrings → canonical version has no docstrings."""
        code = 'def f():\n    """My docstring."""\n    return 1'
        canonical = LLMMutationOperator._canonicalize_code(code)
        assert '"""' not in canonical
        assert "return 1" in canonical

    def test_syntax_error_returns_original(self):
        """Invalid syntax code → returns original unchanged."""
        bad_code = "def f(:\n    return 1"
        result = LLMMutationOperator._canonicalize_code(bad_code)
        assert result == bad_code

    def test_removes_comments(self):
        """Code with # comments → AST unparse drops them."""
        code = "def f():\n    # this is a comment\n    return 1"
        canonical = LLMMutationOperator._canonicalize_code(code)
        assert "# this is a comment" not in canonical
        assert "return 1" in canonical


# ---------------------------------------------------------------------------
# TestOnProgramIngested
# ---------------------------------------------------------------------------


class TestOnProgramIngested:
    async def test_calls_on_mutation_outcome(self):
        """Program with parent IDs → on_mutation_outcome called."""
        agent = AsyncMock()
        llm = MagicMock()
        llm.get_last_model.return_value = "test"
        llm.on_mutation_outcome = MagicMock()
        op = _make_operator(agent, llm_mock=llm)

        parent = _prog()
        child = _prog()
        child.lineage.parents = [parent.id]

        storage = AsyncMock()
        storage.mget.return_value = [parent]

        await op.on_program_ingested(child, storage, outcome=None)

        llm.on_mutation_outcome.assert_called_once()
        call_args = llm.on_mutation_outcome.call_args
        assert call_args[0][0] is child
        assert call_args[0][1] == [parent]

    async def test_no_parents_returns_early(self):
        """Program with no parents → storage.mget not called."""
        agent = AsyncMock()
        llm = MagicMock()
        llm.get_last_model.return_value = "test"
        llm.on_mutation_outcome = MagicMock()
        op = _make_operator(agent, llm_mock=llm)

        child = _prog()
        child.lineage.parents = []

        storage = AsyncMock()

        await op.on_program_ingested(child, storage, outcome=None)

        storage.mget.assert_not_called()
        llm.on_mutation_outcome.assert_not_called()

    async def test_filters_none_parents(self):
        """storage.mget returns [prog, None] → on_mutation_outcome called with [prog] only."""
        agent = AsyncMock()
        llm = MagicMock()
        llm.get_last_model.return_value = "test"
        llm.on_mutation_outcome = MagicMock()
        op = _make_operator(agent, llm_mock=llm)

        parent = _prog()
        child = _prog()
        child.lineage.parents = [parent.id, "deleted-id"]

        storage = AsyncMock()
        storage.mget.return_value = [parent, None]

        await op.on_program_ingested(child, storage, outcome=None)

        llm.on_mutation_outcome.assert_called_once()
        call_args = llm.on_mutation_outcome.call_args
        # Should only have the non-None parent
        assert call_args[0][1] == [parent]
