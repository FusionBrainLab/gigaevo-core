"""Tests for how gigaevo core machinery interacts with the memory system.

Tests the mutate_single → MemorySelectorAgent pipeline and memory metadata flow.
Memory instructions are now injected via the DAG pipeline (MemoryContextStage),
not via explicit parameters on generate_mutations.
"""

from unittest.mock import MagicMock

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.programs.program import Program

# ===========================================================================
# generate_mutations basics
# ===========================================================================


class TestGenerateMutationsMemoryFlow:
    """Test generate_mutations edge cases."""

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
