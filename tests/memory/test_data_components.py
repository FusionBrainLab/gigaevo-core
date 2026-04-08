"""Tests for ideas_tracker data helpers: normalize_improvement_item,
normalize_improvements.

Pure data helpers — no external dependencies.

NOTE: RecordBank, RecordCardExtended, RecordListV2, IncomingIdeas, and the old
ProgramRecord were removed in the ideas-tracker refactor. Their tests have been
deleted along with the source code they tested.
The replacement types are: IdeaBank (idea_bank.py), Idea/ProgramRecord (models.py).
"""

import pytest

from gigaevo.memory.ideas_tracker.models import (
    UsageEntry,
    UsagePayload,
    normalize_improvement_item,
    normalize_improvements,
)

# ===========================================================================
# normalize_improvement_item
# ===========================================================================


class TestNormalizeImprovementItem:
    def test_string_input(self):
        result = normalize_improvement_item("Use SA for local search")
        assert result["description"] == "Use SA for local search"
        assert result["explanation"] == ""

    def test_dict_with_description_and_explanation(self):
        result = normalize_improvement_item(
            {
                "description": "SA refinement",
                "explanation": "Improves convergence",
            }
        )
        assert result["description"] == "SA refinement"
        assert result["explanation"] == "Improves convergence"

    def test_dict_with_alternative_keys(self):
        result = normalize_improvement_item(
            {
                "summary": "SA method",
                "rationale": "Better convergence",
            }
        )
        assert result["description"] == "SA method"
        assert result["explanation"] == "Better convergence"

    def test_non_dict_non_string(self):
        result = normalize_improvement_item(42)
        assert result["description"] == "42"

    def test_none_input(self):
        result = normalize_improvement_item(None)
        assert result["description"] == "Unspecified change"

    def test_empty_dict(self):
        result = normalize_improvement_item({})
        assert result["description"] == "Unspecified change"

    def test_dict_with_only_unknown_keys(self):
        result = normalize_improvement_item({"custom_field": "value"})
        assert "custom_field: value" in result["description"]

    def test_nested_dict_stringified(self):
        result = normalize_improvement_item(
            {
                "description": {"nested": "value", "other": "data"},
            }
        )
        assert "nested: value" in result["description"]


class TestNormalizeImprovements:
    def test_none_returns_empty(self):
        assert normalize_improvements(None) == []

    def test_list_of_strings(self):
        result = normalize_improvements(["idea A", "idea B"])
        assert len(result) == 2
        assert result[0]["description"] == "idea A"

    def test_single_value_wrapped(self):
        result = normalize_improvements("single idea")
        assert len(result) == 1
        assert result[0]["description"] == "single idea"


# ===========================================================================
# UsageEntry and UsagePayload
# ===========================================================================


class TestUsageModels:
    def test_usage_entry_fields(self) -> None:
        entry = UsageEntry(
            task_description_summary="Multi-hop QA",
            used_count=3,
            fitness_delta_per_use=[0.05, 0.02, -0.01],
            median_delta_fitness=0.02,
        )
        assert entry.task_description_summary == "Multi-hop QA"
        assert entry.used_count == 3
        assert entry.fitness_delta_per_use == [0.05, 0.02, -0.01]
        assert entry.median_delta_fitness == pytest.approx(0.02)

    def test_usage_entry_none_median(self) -> None:
        entry = UsageEntry(
            task_description_summary="Task",
            used_count=0,
            fitness_delta_per_use=[],
            median_delta_fitness=None,
        )
        assert entry.median_delta_fitness is None

    def test_usage_payload_model(self) -> None:
        payload = UsagePayload(
            entries=[
                UsageEntry(
                    task_description_summary="Task A",
                    used_count=2,
                    fitness_delta_per_use=[0.1, 0.2],
                    median_delta_fitness=0.15,
                )
            ],
            total_used=2,
            median_delta_fitness=0.15,
        )
        assert len(payload.entries) == 1
        assert payload.total_used == 2
        assert payload.median_delta_fitness == pytest.approx(0.15)

    def test_usage_payload_defaults(self) -> None:
        payload = UsagePayload()
        assert payload.entries == []
        assert payload.total_used == 0
        assert payload.median_delta_fitness is None

    def test_usage_payload_serialization_roundtrip(self) -> None:
        payload = UsagePayload(
            entries=[
                UsageEntry(
                    task_description_summary="task",
                    used_count=1,
                    fitness_delta_per_use=[0.5],
                    median_delta_fitness=0.5,
                )
            ],
            total_used=1,
            median_delta_fitness=0.5,
        )
        data = payload.model_dump()
        restored = UsagePayload.model_validate(data)
        assert restored.total_used == 1
        assert restored.entries[0].task_description_summary == "task"
