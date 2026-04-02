"""Integration tests: LLMMutationOperator with REAL constructor + memory.

Uses the same factory-patching pattern as test_mutation_operator.py,
but focuses on the memory pipeline: MemorySelectorAgent wiring, memory
card injection into mutation prompts, and the full mutate_single flow
with memory_instructions.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.evolution.mutation.context import (
    MUTATION_CONTEXT_METADATA_KEY,
    MUTATION_MEMORY_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator
from gigaevo.llm.agents.memory_selector import MemorySelection, MemorySelectorAgent
from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_problem_context():
    ctx = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="accuracy",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        }
    )
    pc = MagicMock()
    pc.task_description = "Multi-hop fact verification"
    pc.metrics_context = ctx
    return pc


def _make_llm_mock():
    llm = MagicMock()
    llm.get_last_model.return_value = "test-model"
    llm.on_mutation_outcome = MagicMock()
    return llm


def _make_agent_mock(code="def solve(): return 42"):
    """Agent that returns a dict matching the real agent's arun output."""
    agent = AsyncMock()
    agent.arun.return_value = {
        "code": code,
        "raw_output": "mutated code",
        "model_used": "test-model",
        "structured_output": None,
    }
    return agent


def _make_selector_mock(cards=None, card_ids=None):
    """MemorySelectorAgent mock returning pre-defined cards."""
    selector = AsyncMock(spec=MemorySelectorAgent)
    selector.select.return_value = MemorySelection(
        cards=cards or [],
        card_ids=card_ids or [],
    )
    return selector


def _make_operator(agent_mock, selector_mock=None, **kwargs):
    """Build LLMMutationOperator with REAL constructor, patched factories."""
    llm = _make_llm_mock()

    with patch(
        "gigaevo.evolution.mutation.mutation_operator.create_mutation_agent",
        return_value=agent_mock,
    ), patch(
        "gigaevo.evolution.mutation.mutation_operator.create_memory_selector_agent",
        return_value=selector_mock or _make_selector_mock(),
    ):
        op = LLMMutationOperator(
            llm_wrapper=llm,
            problem_context=_make_problem_context(),
            mutation_mode=kwargs.get("mode", "rewrite"),
            strip_comments_and_docstrings=kwargs.get("strip", False),
        )
    return op


def _make_program(code="def solve(): return 1", **metadata):
    p = Program(code=code, state=ProgramState.DONE)
    p.metadata.update(metadata)
    return p


# ===========================================================================
# 1. Real constructor wiring
# ===========================================================================


class TestOperatorConstructorWiring:
    """Verify real __init__ wires memory_selector correctly."""

    def test_constructor_creates_memory_selector(self):
        agent = _make_agent_mock()
        selector = _make_selector_mock()
        op = _make_operator(agent, selector)

        assert op.memory_selector is selector
        assert op.agent is agent
        assert op.mutation_mode == "rewrite"

    def test_constructor_creates_metrics_formatter(self):
        op = _make_operator(_make_agent_mock())
        desc = op.metrics_formatter.format_metrics_description()
        assert isinstance(desc, str)
        assert "fitness" in desc.lower() or "accuracy" in desc.lower()


# ===========================================================================
# 2. mutate_single with memory: real constructor, mocked agents
# ===========================================================================


class TestMutateSingleWithMemory:
    """Full mutate_single flow with memory_instructions via real constructor."""

    @pytest.mark.asyncio
    async def test_memory_cards_injected_into_parent_metadata(self):
        """When selector returns cards, they appear in parent.metadata."""
        captured_input = {}

        async def capturing_arun(*, input, mutation_mode):
            captured_input["parents"] = input
            captured_input["mode"] = mutation_mode
            return {
                "code": "def solve(): return 42",
                "raw_output": "ok",
                "model_used": "test",
                "structured_output": None,
            }

        agent = MagicMock()
        agent.arun = capturing_arun

        selector = _make_selector_mock(
            cards=["1. Sort evidence by relevance", "2. Filter low-confidence hops"],
            card_ids=["idea-1", "idea-2"],
        )
        op = _make_operator(agent, selector)

        parent = _make_program()
        result = await op.mutate_single([parent], memory_instructions="use memory")

        assert result is not None
        assert result.code == "def solve(): return 42"

        # Verify memory card IDs propagated to MutationSpec output
        assert MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY in result.metadata
        assert result.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == ["idea-1", "idea-2"]

        # Verify selector was called
        selector.select.assert_awaited_once()
        call_kwargs = selector.select.call_args.kwargs
        assert call_kwargs["mutation_mode"] == "rewrite"
        assert "Multi-hop fact verification" in call_kwargs["task_description"]
        assert call_kwargs["max_cards"] == 3

        # Verify parents received memory metadata
        mutated_parents = captured_input["parents"]
        assert len(mutated_parents) == 1
        meta = mutated_parents[0].metadata
        assert MUTATION_MEMORY_METADATA_KEY in meta
        assert "Sort evidence" in meta[MUTATION_MEMORY_METADATA_KEY]
        assert "Filter low-confidence" in meta[MUTATION_MEMORY_METADATA_KEY]
        assert meta[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == ["idea-1", "idea-2"]

    @pytest.mark.asyncio
    async def test_original_parent_unchanged_after_memory_injection(self):
        agent = _make_agent_mock()
        selector = _make_selector_mock(
            cards=["1. Use SA"],
            card_ids=["idea-1"],
        )
        op = _make_operator(agent, selector)

        # Use MUTABLE metadata value to test deep copy
        parent = _make_program()
        parent.metadata["nested_list"] = ["original"]
        original_meta_snapshot = {
            k: list(v) if isinstance(v, list) else v
            for k, v in parent.metadata.items()
        }

        await op.mutate_single([parent], memory_instructions="use memory")

        # Original parent must NOT have memory metadata
        assert MUTATION_MEMORY_METADATA_KEY not in parent.metadata
        # Mutable nested values must be independent (deep copy check)
        assert parent.metadata["nested_list"] == ["original"]

    @pytest.mark.asyncio
    async def test_empty_memory_selection_no_injection(self):
        """When selector returns no cards, parents get no memory metadata."""
        captured = {}

        async def spy_arun(*, input, mutation_mode):
            captured["parents"] = input
            return {"code": "def f(): pass", "raw_output": "", "model_used": "t",
                    "structured_output": None}

        agent = MagicMock()
        agent.arun = spy_arun
        selector = _make_selector_mock(cards=[], card_ids=[])
        op = _make_operator(agent, selector)

        parent = _make_program()
        await op.mutate_single([parent], memory_instructions="use memory")

        # No cards → no memory metadata injection
        meta = captured["parents"][0].metadata
        assert MUTATION_MEMORY_METADATA_KEY not in meta

    @pytest.mark.asyncio
    async def test_no_memory_instructions_skips_selector(self):
        agent = _make_agent_mock()
        selector = _make_selector_mock()
        op = _make_operator(agent, selector)

        parent = _make_program()
        await op.mutate_single([parent], memory_instructions=None)

        selector.select.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiple_parents_all_get_memory(self):
        """With 2 parents, both get memory metadata clones."""
        captured = {}

        async def spy(*, input, mutation_mode):
            captured["parents"] = input
            return {"code": "def f(): pass", "raw_output": "", "model_used": "t",
                    "structured_output": None}

        agent = MagicMock()
        agent.arun = spy
        selector = _make_selector_mock(
            cards=["1. SA optimization"], card_ids=["idea-1"],
        )
        op = _make_operator(agent, selector)

        p1 = _make_program(code="def a(): return 1")
        p2 = _make_program(code="def b(): return 2")
        await op.mutate_single([p1, p2], memory_instructions="use memory")

        parents = captured["parents"]
        assert len(parents) == 2
        for p in parents:
            assert MUTATION_MEMORY_METADATA_KEY in p.metadata
            assert "SA optimization" in p.metadata[MUTATION_MEMORY_METADATA_KEY]


# ===========================================================================
# 3. MemorySelectorAgent with real AmemGamMemory
# ===========================================================================


class TestSelectorWithRealMemory:
    """Wire MemorySelectorAgent with pre-filled local AmemGamMemory."""

    def _make_selector(self, tmp_path, ideas):
        mem = AmemGamMemory(
            checkpoint_path=str(tmp_path / "mem"),
            use_api=False, sync_on_init=False,
            enable_llm_synthesis=False, enable_memory_evolution=False,
            enable_llm_card_enrichment=False,
        )
        for idea in ideas:
            mem.save_card(idea)

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = None
        selector.memory = mem
        return selector

    @pytest.mark.asyncio
    async def test_search_returns_relevant_cards(self, tmp_path):
        selector = self._make_selector(tmp_path, [
            {"id": "idea-1",
             "description": "Sort evidence by relevance score for multi-hop verification",
             "keywords": ["sort", "relevance", "evidence", "verification", "multi"],
             "task_description": "Multi-hop fact verification"},
            {"id": "idea-2",
             "description": "Filter low-confidence hops using threshold for fact checking",
             "keywords": ["filter", "confidence", "fact", "verification"],
             "task_description": "Multi-hop fact verification"},
        ])
        parent = _make_program(code="def solve(x):\n    return x\n")

        selection = await selector.select(
            input=[parent],
            mutation_mode="rewrite",
            task_description="Multi-hop fact verification",
            metrics_description="fitness: accuracy",
            memory_text="",
            max_cards=3,
        )

        assert len(selection.cards) > 0

    @pytest.mark.asyncio
    async def test_build_request_contains_parent_code(self, tmp_path):
        selector = self._make_selector(tmp_path, [])
        parent = _make_program(code="def solve(x):\n    return sorted(x)\n")

        query = selector._build_request(
            parents=[parent],
            mutation_mode="rewrite",
            task_description="Multi-hop fact verification",
            metrics_description="fitness: accuracy on validation set",
            max_cards=3,
        )

        assert "MUTATION INPUTS" in query
        assert "TASK DESCRIPTION:" in query
        assert "Multi-hop fact verification" in query
        assert "AVAILABLE METRICS:" in query
        assert "accuracy on validation set" in query
        assert "MUTATION MODE:" in query
        assert "rewrite" in query
        assert "def solve(x):" in query
        assert "return sorted(x)" in query
        assert "Return exactly 3 concise ideas" in query

    @pytest.mark.asyncio
    async def test_build_request_includes_mutation_context(self, tmp_path):
        """Parent with mutation_context metadata → appears in request."""
        selector = self._make_selector(tmp_path, [])
        parent = _make_program(code="def f(): pass")
        parent.metadata[MUTATION_CONTEXT_METADATA_KEY] = "Previous mutation improved sorting"

        query = selector._build_request(
            parents=[parent],
            mutation_mode="diff",
            task_description="test task",
            metrics_description="fitness",
            max_cards=1,
        )

        assert "Previous mutation improved sorting" in query
        assert "diff" in query

    @pytest.mark.asyncio
    async def test_search_with_ids_extracts_card_ids(self, tmp_path):
        """_search_with_ids returns card IDs from search output."""
        selector = self._make_selector(tmp_path, [
            {"id": "idea-abc-123", "description": "Use simulated annealing",
             "keywords": ["annealing"]},
        ])

        text, ids = selector._search_with_ids("simulated annealing")

        assert "idea-abc-123" in text
        assert "idea-abc-123" in ids

    @pytest.mark.asyncio
    async def test_parse_search_result_formats(self, tmp_path):
        selector = self._make_selector(tmp_path, [])

        # Numbered list
        text1 = "1. Sort by relevance\n2. Filter noise\n3. Limit depth\n"
        cards1 = selector._parse_search_result(text1, max_cards=2)
        assert len(cards1) == 2
        assert "Sort by relevance" in cards1[0]

        # Bulleted list
        text2 = "- Sort by relevance\n- Filter noise\n"
        cards2 = selector._parse_search_result(text2, max_cards=5)
        assert len(cards2) == 2

        # No relevant memories
        text3 = "No relevant memories found."
        cards3 = selector._parse_search_result(text3, max_cards=5)
        assert cards3 == []

        # Plain text (no format)
        text4 = "Use simulated annealing for local search"
        cards4 = selector._parse_search_result(text4, max_cards=3)
        assert len(cards4) == 1
        assert "simulated annealing" in cards4[0]

    @pytest.mark.asyncio
    async def test_extract_card_ids_from_formatted_output(self, tmp_path):
        selector = self._make_selector(tmp_path, [])

        # Standard format from _format_search_results
        text = (
            "Query: annealing\n\n"
            "Top relevant memory cards:\n"
            "1. idea-abc [general] Use simulated annealing\n"
            "2. idea-def [retrieval] Filter noise\n"
        )
        ids = selector._extract_card_ids_from_text(text)
        assert "idea-abc" in ids
        assert "idea-def" in ids

    @pytest.mark.asyncio
    async def test_merge_card_ids_deduplicates_and_limits(self, tmp_path):
        selector = self._make_selector(tmp_path, [])

        merged = selector._merge_card_ids(
            primary=["a", "b", "c"],
            secondary=["b", "d", "e"],
            max_cards=3,
        )
        assert merged == ["a", "b", "c"]  # Limited to max_cards=3

        merged_all = selector._merge_card_ids(
            primary=["a"], secondary=["b", "c"], max_cards=10,
        )
        assert merged_all == ["a", "b", "c"]
