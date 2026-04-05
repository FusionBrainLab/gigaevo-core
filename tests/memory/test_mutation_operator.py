"""Integration tests: MemorySelectorAgent with real AmemGamMemory.

Memory injection into mutation prompts is now handled by the DAG pipeline
(MemoryContextStage → MutationContextStage), not by LLMMutationOperator.
These tests verify the MemorySelectorAgent search/parse/ID-extraction logic.
"""

from __future__ import annotations

import asyncio

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_CONTEXT_METADATA_KEY,
)
from gigaevo.llm.agents.memory_selector import MemorySelectorAgent
from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_program(code="def solve(): return 1", **metadata):
    p = Program(code=code, state=ProgramState.DONE)
    p.metadata.update(metadata)
    return p


# ===========================================================================
# MemorySelectorAgent with real AmemGamMemory
# ===========================================================================


class TestSelectorWithRealMemory:
    """Wire MemorySelectorAgent with pre-filled local AmemGamMemory."""

    def _make_selector(self, tmp_path, ideas):
        mem = AmemGamMemory(
            checkpoint_path=str(tmp_path / "mem"),
            use_api=False,
            sync_on_init=False,
            enable_llm_synthesis=False,
            enable_memory_evolution=False,
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
        selector = self._make_selector(
            tmp_path,
            [
                {
                    "id": "idea-1",
                    "description": "Sort evidence by relevance score for multi-hop verification",
                    "keywords": [
                        "sort",
                        "relevance",
                        "evidence",
                        "verification",
                        "multi",
                    ],
                    "task_description": "Multi-hop fact verification",
                },
                {
                    "id": "idea-2",
                    "description": "Filter low-confidence hops using threshold for fact checking",
                    "keywords": ["filter", "confidence", "fact", "verification"],
                    "task_description": "Multi-hop fact verification",
                },
            ],
        )
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
        parent.metadata[MUTATION_CONTEXT_METADATA_KEY] = (
            "Previous mutation improved sorting"
        )

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
        selector = self._make_selector(
            tmp_path,
            [
                {
                    "id": "idea-abc-123",
                    "description": "Use simulated annealing",
                    "keywords": ["annealing"],
                },
            ],
        )

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
            primary=["a"],
            secondary=["b", "c"],
            max_cards=10,
        )
        assert merged_all == ["a", "b", "c"]
