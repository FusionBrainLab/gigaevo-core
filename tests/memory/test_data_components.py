"""Tests for ideas_tracker data helpers: normalize_improvement_item,
normalize_improvements.

Pure data helpers — no external dependencies.

NOTE: RecordBank, RecordCardExtended, RecordListV2, IncomingIdeas, and the old
ProgramRecord were removed in the ideas-tracker refactor. Their tests have been
deleted along with the source code they tested.
The replacement types are: IdeaBank (idea_bank.py), Idea/ProgramRecord (models.py).
"""


from gigaevo.memory.ideas_tracker.models import (
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
