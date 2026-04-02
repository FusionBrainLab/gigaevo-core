"""Tests for how gigaevo core machinery interacts with the memory system.

Tests the generate_mutations → mutate_single → MemorySelectorAgent pipeline,
and the EvolutionEngine.memory_enabled flag behavior.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.mutation.context import (
    MUTATION_CONTEXT_METADATA_KEY,
    MUTATION_MEMORY_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.programs.program import Program

# ===========================================================================
# generate_mutations with memory_instructions
# ===========================================================================


class TestGenerateMutationsMemoryFlow:
    """Test the generate_mutations function's memory_instructions path."""

    @pytest.mark.asyncio
    async def test_memory_instructions_passed_to_mutate_single(self):
        """When memory_instructions is not None, it flows to mutate_single."""
        from gigaevo.evolution.engine.mutation import generate_mutations

        # Track what mutate_single receives
        captured_calls = []

        async def mock_mutate_single(parents, memory_instructions=None):
            captured_calls.append(
                {
                    "parents": parents,
                    "memory_instructions": memory_instructions,
                }
            )
            # Return a valid MutationSpec-like object
            from gigaevo.evolution.mutation.base import MutationSpec

            return MutationSpec(
                code="def solve(): return 1",
                parents=[p.id for p in parents],
            )

        mock_mutator = MagicMock()
        mock_mutator.mutate_single = mock_mutate_single

        mock_storage = AsyncMock()
        mock_storage.add = AsyncMock(return_value="new-prog-id")

        mock_state = AsyncMock()

        mock_selector = MagicMock()
        mock_selector.create_parent_iterator.return_value = iter(
            [[Program(code="def f(): pass", metadata={})]]
        )

        await generate_mutations(
            [Program(code="def f(): pass", metadata={})],
            mutator=mock_mutator,
            storage=mock_storage,
            state_manager=mock_state,
            parent_selector=mock_selector,
            limit=1,
            iteration=5,
            memory_instructions="Use SA for optimization",
            memory_used=True,
        )

        assert len(captured_calls) == 1
        assert captured_calls[0]["memory_instructions"] == "Use SA for optimization"

    @pytest.mark.asyncio
    async def test_no_memory_instructions_passed_as_none(self):
        """When memory_instructions is None, mutate_single gets None or not called with it."""
        from gigaevo.evolution.engine.mutation import generate_mutations

        captured_calls = []

        async def mock_mutate_single(parents, memory_instructions=None):
            captured_calls.append({"memory_instructions": memory_instructions})
            from gigaevo.evolution.mutation.base import MutationSpec

            return MutationSpec(
                code="def solve(): return 1",
                parents=[p.id for p in parents],
            )

        mock_mutator = MagicMock()
        mock_mutator.mutate_single = mock_mutate_single

        mock_storage = AsyncMock()
        mock_storage.add = AsyncMock(return_value="new-prog-id")
        mock_state = AsyncMock()
        mock_selector = MagicMock()
        mock_selector.create_parent_iterator.return_value = iter(
            [[Program(code="def f(): pass", metadata={})]]
        )

        await generate_mutations(
            [Program(code="def f(): pass", metadata={})],
            mutator=mock_mutator,
            storage=mock_storage,
            state_manager=mock_state,
            parent_selector=mock_selector,
            limit=1,
            iteration=5,
            memory_instructions=None,
            memory_used=False,
        )

        assert len(captured_calls) == 1
        assert captured_calls[0]["memory_instructions"] is None

    @pytest.mark.asyncio
    async def test_empty_elites_returns_empty(self):
        from gigaevo.evolution.engine.mutation import generate_mutations

        result = await generate_mutations(
            [],
            mutator=MagicMock(),
            storage=MagicMock(),
            state_manager=MagicMock(),
            parent_selector=MagicMock(),
            limit=5,
            iteration=1,
        )
        assert result == []


# ===========================================================================
# LLMMutationOperator end-to-end memory injection
# ===========================================================================


class TestMutationOperatorMemoryInjection:
    """Test the full memory injection path in LLMMutationOperator."""

    def _make_operator(self, memory_cards=None):
        """Create LLMMutationOperator with mocked internals and real memory."""
        from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator
        from gigaevo.llm.agents.memory_selector import MemorySelection

        operator = LLMMutationOperator.__new__(LLMMutationOperator)
        operator.mutation_mode = "rewrite"
        operator.fallback_to_rewrite = True
        operator.context_key = MUTATION_CONTEXT_METADATA_KEY
        operator.problem_context = MagicMock()
        operator.problem_context.task_description = "Multi-hop fact verification"
        operator.metrics_formatter = MagicMock()
        operator.metrics_formatter.format_metrics_description.return_value = (
            "fitness: accuracy"
        )
        operator.prompt_fetcher = None
        operator.storage = None
        operator.bandit = None
        operator.strip_comments_and_docstrings = False
        operator.llm_wrapper = MagicMock()
        operator.llm_wrapper.get_last_model.return_value = "test-model"

        # Mock memory selector
        cards = memory_cards or ["1. Sort evidence by relevance score"]
        mock_selector = AsyncMock()
        mock_selector.select.return_value = MemorySelection(
            cards=cards,
            card_ids=["idea-1"],
        )
        operator.memory_selector = mock_selector

        # Mock mutation agent
        captured = {"parents": None}

        async def mock_arun(*, input, mutation_mode):
            captured["parents"] = input
            return {
                "code": "def solve(x): return sorted(x)",
                "raw_output": "ok",
                "model_used": "test",
                "structured_output": None,
            }

        operator.agent = MagicMock()
        operator.agent.arun = mock_arun

        return operator, captured

    @pytest.mark.asyncio
    async def test_memory_cards_in_parent_metadata(self):
        """When memory_instructions is provided, cards appear in parent metadata."""
        operator, captured = self._make_operator()
        parent = Program(code="def f(): return 1", metadata={})

        result = await operator.mutate_single(
            [parent], memory_instructions="use memory"
        )
        assert result is not None

        # The mutation agent received parents WITH memory metadata
        mutated_parent = captured["parents"][0]
        assert MUTATION_MEMORY_METADATA_KEY in mutated_parent.metadata
        assert "Sort evidence" in mutated_parent.metadata[MUTATION_MEMORY_METADATA_KEY]
        assert mutated_parent.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == [
            "idea-1"
        ]

    @pytest.mark.asyncio
    async def test_original_parent_not_mutated(self):
        """Deep copy: original parent's metadata is unchanged."""
        operator, _ = self._make_operator()
        parent = Program(code="def f(): return 1", metadata={"existing": "data"})

        await operator.mutate_single([parent], memory_instructions="use memory")

        assert MUTATION_MEMORY_METADATA_KEY not in parent.metadata
        assert parent.metadata == {"existing": "data"}

    @pytest.mark.asyncio
    async def test_empty_memory_selection_skips_injection(self):
        """When memory search returns no cards, parents get no memory metadata."""
        from gigaevo.llm.agents.memory_selector import MemorySelection

        operator, captured = self._make_operator(memory_cards=[])
        # Override to return empty
        operator.memory_selector.select.return_value = MemorySelection(
            cards=[],
            card_ids=[],
        )

        parent = Program(code="def f(): return 1", metadata={})
        await operator.mutate_single([parent], memory_instructions="use memory")

        # Parents should NOT have memory metadata (empty selection)
        mutated_parent = captured["parents"][0]
        assert MUTATION_MEMORY_METADATA_KEY not in mutated_parent.metadata

    @pytest.mark.asyncio
    async def test_multiple_memory_cards_joined(self):
        """Multiple cards are joined with double newline."""
        operator, captured = self._make_operator(
            memory_cards=["1. Sort by relevance", "2. Filter noise", "3. Limit depth"]
        )
        parent = Program(code="def f(): return 1", metadata={})

        await operator.mutate_single([parent], memory_instructions="use memory")

        memory_block = captured["parents"][0].metadata[MUTATION_MEMORY_METADATA_KEY]
        assert "Sort by relevance" in memory_block
        assert "Filter noise" in memory_block
        assert "Limit depth" in memory_block

    @pytest.mark.asyncio
    async def test_mutation_returns_valid_spec(self):
        """The mutation result has code, parent_ids, and metadata."""
        operator, _ = self._make_operator()
        parent = Program(code="def f(): return 1", metadata={})

        result = await operator.mutate_single(
            [parent], memory_instructions="use memory"
        )
        assert result is not None
        assert result.code == "def solve(x): return sorted(x)"
        assert len(result.parents) == 1


# ===========================================================================
# Program metadata after mutation round-trip
# ===========================================================================


class TestProgramMetadataRoundtrip:
    """Test that memory metadata survives the full mutation pipeline."""

    def test_program_preserves_memory_metadata_on_parent(self):
        """Parent program can carry memory metadata that the mutation agent sees."""
        parent = Program(
            code="def f(): return 1",
            metadata={
                MUTATION_MEMORY_METADATA_KEY: "1. Sort by relevance",
                MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY: ["idea-1", "idea-2"],
            },
        )

        # Verify the metadata is accessible
        assert parent.metadata[MUTATION_MEMORY_METADATA_KEY] == "1. Sort by relevance"
        assert parent.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == [
            "idea-1",
            "idea-2",
        ]

        # Deep copy preserves metadata
        clone = parent.model_copy(deep=True)
        assert clone.metadata[MUTATION_MEMORY_METADATA_KEY] == "1. Sort by relevance"
        # Modifying clone doesn't affect original
        clone.metadata["extra"] = "test"
        assert "extra" not in parent.metadata

    def test_mutation_spec_metadata_key_constants(self):
        """MutationSpec metadata key constants are accessible."""
        from gigaevo.evolution.mutation.base import MutationSpec

        assert hasattr(MutationSpec, "META_MODEL")
        assert hasattr(MutationSpec, "META_OUTPUT")
        assert hasattr(MutationSpec, "META_PROMPT_ID")
