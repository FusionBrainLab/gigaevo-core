"""End-to-end integration: memory selector in the mutation loop.

Memory instructions are now injected via the DAG pipeline (MemoryContextStage),
not via explicit engine config flags. This file tests the MemorySelectorAgent
component that provides memory cards to the pipeline.
"""

from __future__ import annotations

import asyncio

import pytest

from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.programs.program import Program
from tests.fakes.agentic_memory import make_test_memory


def _make_memory(tmp_path, **overrides) -> AmemGamMemory:
    return make_test_memory(tmp_path, **overrides)


# ===========================================================================
# Memory selector in the mutation loop
# ===========================================================================


class TestMemorySelectorInMutationLoop:
    """Wire MemorySelectorAgent with real memory into the mutation flow."""

    @pytest.mark.asyncio
    async def test_selector_returns_cards_from_memory(self, tmp_path) -> None:
        """MemorySelectorAgent.select() returns cards from pre-filled memory."""
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        mem = _make_memory(tmp_path)
        mem.save_card(
            {
                "id": "idea-1",
                "description": "Sort evidence by relevance score for better chain quality",
                "keywords": ["sort", "relevance", "evidence", "chain"],
            }
        )
        mem.save_card(
            {
                "id": "idea-2",
                "description": "Filter low-confidence hops using threshold",
                "keywords": ["filter", "confidence", "threshold"],
            }
        )

        # Create selector with injected memory
        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = None
        selector.memory = mem

        parent = Program(
            code="def solve(x):\n    return x\n",
            metadata={},
        )

        selection = await selector.select(
            input=[parent],
            mutation_mode="rewrite",
            task_description="Multi-hop fact verification",
            metrics_description="fitness: accuracy on validation set",
            memory_text="",
            max_cards=3,
        )

        # Should find relevant cards
        assert len(selection.cards) > 0, (
            "Selector returned no cards from pre-filled memory"
        )

        # Card IDs should be extractable
        assert isinstance(selection.card_ids, list)

    @pytest.mark.asyncio
    async def test_selector_with_empty_memory_returns_empty(self, tmp_path) -> None:
        """Selector with no cards returns empty selection."""
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        mem = _make_memory(tmp_path)  # Empty

        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = None
        selector.memory = mem

        selection = await selector.select(
            input=[Program(code="def f(): pass", metadata={})],
            mutation_mode="rewrite",
            task_description="test",
            metrics_description="fitness",
            memory_text="",
            max_cards=3,
        )

        # Empty memory → "No relevant memories" → no cards parsed
        assert selection.cards == []
