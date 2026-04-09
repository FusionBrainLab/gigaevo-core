"""End-to-end tests for the complete memory system flow with usage tracking.

Tests the full cycle: idea creation → usage application → card serialization
→ JSON round-trip → card deserialization, verifying data integrity at each step.
"""

import json

from gigaevo.memory.ideas_tracker.idea_bank import (
    IdeaBank,
    build_usage_payload,
    merge_usage_payloads,
)
from gigaevo.memory.ideas_tracker.models import Idea
from gigaevo.memory.shared_memory.card_conversion import (
    card_to_concept_content,
    concept_to_card,
    normalize_memory_card,
)


class TestE2EMemoryFlow:
    """Integration tests for the complete memory system flow."""

    def test_idea_to_card_with_usage_roundtrip(self) -> None:
        """Complete flow: create idea → apply usage → serialize → deserialize."""
        # Step 1: Create idea
        idea = Idea(
            id="idea-001",
            description="Optimize loop structure",
            task_description="Loop optimization task",
            task_description_summary="loop-opt",
        )

        # Step 2: Build and apply usage
        usage1 = build_usage_payload(
            {
                "task-alpha": [0.5, 0.6],
                "task-beta": [0.3],
            }
        )
        idea_with_usage = idea.model_copy(update={"usage": usage1})

        # Step 3: Store in bank and merge additional usage
        bank = IdeaBank()
        bank.add(idea_with_usage)

        usage2 = build_usage_payload(
            {
                "task-alpha": [0.2],
                "task-gamma": [0.8, 0.9],
            }
        )
        bank.apply_usage_updates({idea.id: usage2})
        updated_idea = bank.get(idea.id)

        # Step 4: Convert to card dict (serialization)
        card_dict = card_to_concept_content(
            card=normalize_memory_card(
                {
                    "id": idea.id,
                    "description": idea.description,
                    "task_description": idea.task_description,
                    "task_description_summary": idea.task_description_summary,
                    "usage": updated_idea.usage.model_dump(),
                }
            )
        )

        # Verify serialized usage is a dict
        assert isinstance(card_dict["usage"], dict)
        assert card_dict["usage"]["total_used"] == 6
        assert len(card_dict["usage"]["entries"]) == 3

        # Step 5: JSON round-trip
        json_str = json.dumps(card_dict)
        card_dict_restored = json.loads(json_str)

        # Step 6: Deserialize back to card
        card_restored = concept_to_card(card_dict_restored, fallback_id="fallback-123")

        # Step 7: Verify data integrity
        assert card_restored.usage.total_used == 6
        assert len(card_restored.usage.entries) == 3
        # Verify task entries
        task_names = {e.task_description_summary for e in card_restored.usage.entries}
        assert task_names == {"task-alpha", "task-beta", "task-gamma"}

    def test_usage_merge_with_dict_input(self) -> None:
        """Test merging dict-based usage (as in write_pipeline)."""
        dict_usage_1 = {
            "entries": [
                {
                    "task_description_summary": "task-1",
                    "used_count": 1,
                    "fitness_delta_per_use": [0.1],
                }
            ],
            "total_used": 1,
            "median_delta_fitness": 0.1,
        }
        dict_usage_2 = {
            "entries": [
                {
                    "task_description_summary": "task-1",
                    "used_count": 1,
                    "fitness_delta_per_use": [0.2],
                }
            ],
            "total_used": 1,
            "median_delta_fitness": 0.2,
        }

        merged = merge_usage_payloads(dict_usage_1, dict_usage_2)
        assert merged.total_used == 2
        assert len(merged.entries) == 1
        assert merged.entries[0].fitness_delta_per_use == [0.1, 0.2]

        # Verify round-trip through model_dump
        merged_dict = merged.model_dump()
        assert merged_dict["total_used"] == 2
