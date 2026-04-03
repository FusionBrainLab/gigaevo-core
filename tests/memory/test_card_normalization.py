"""Tests for normalize_memory_card and its helper functions.

Pin down the exact normalization behavior so refactoring can be validated.
"""

from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.models import (
    MemoryCard,
    MemoryCardExplanation,
    ProgramCard,
)
from gigaevo.memory.shared_memory.utils import _to_float, _to_int, _to_list

# ===========================================================================
# _to_list
# ===========================================================================


class TestToList:
    def test_list_passthrough(self):
        assert _to_list([1, 2]) == [1, 2]

    def test_empty_list(self):
        assert _to_list([]) == []

    def test_none_returns_empty(self):
        assert _to_list(None) == []

    def test_scalar_wrapped(self):
        assert _to_list("hello") == ["hello"]

    def test_int_wrapped(self):
        assert _to_list(42) == [42]

    def test_dict_wrapped(self):
        d = {"a": 1}
        assert _to_list(d) == [d]

    def test_nested_list_not_flattened(self):
        assert _to_list([[1, 2]]) == [[1, 2]]


# ===========================================================================
# _to_int
# ===========================================================================


class TestToInt:
    def test_valid_int(self):
        assert _to_int(5) == 5

    def test_valid_string(self):
        assert _to_int("10") == 10

    def test_float_truncates(self):
        assert _to_int(3.9) == 3

    def test_invalid_returns_default(self):
        assert _to_int("abc") == 0

    def test_invalid_custom_default(self):
        assert _to_int("abc", default=-1) == -1

    def test_none_returns_default(self):
        assert _to_int(None) == 0

    def test_empty_string(self):
        assert _to_int("") == 0


# ===========================================================================
# _to_float
# ===========================================================================


class TestToFloat:
    def test_valid_float(self):
        assert _to_float(3.14) == 3.14

    def test_valid_string(self):
        assert _to_float("2.5") == 2.5

    def test_int_promoted(self):
        assert _to_float(7) == 7.0

    def test_invalid_returns_default_none(self):
        assert _to_float("abc") is None

    def test_invalid_custom_default(self):
        assert _to_float("abc", default=0.0) == 0.0

    def test_none_returns_default(self):
        assert _to_float(None) is None

    def test_empty_string(self):
        assert _to_float("") is None

    def test_negative(self):
        assert _to_float("-1.5") == -1.5

    def test_inf(self):
        import math

        assert math.isinf(_to_float("inf"))

    def test_nan(self):
        import math

        assert math.isnan(_to_float("nan"))


# ===========================================================================
# normalize_memory_card — general cards
# ===========================================================================


class TestNormalizeGeneralCard:
    def test_none_input(self):
        result = normalize_memory_card(None)
        assert isinstance(result, MemoryCard)
        assert result.id == ""
        assert result.category == "general"

    def test_empty_dict(self):
        result = normalize_memory_card({})
        assert isinstance(result, MemoryCard)
        assert result.description == ""
        assert result.programs == []
        assert result.keywords == []

    def test_fallback_id(self):
        result = normalize_memory_card({}, fallback_id="fb-1")
        assert result.id == "fb-1"

    def test_id_in_card_overrides_fallback(self):
        result = normalize_memory_card({"id": "card-1"}, fallback_id="fb-1")
        assert result.id == "card-1"

    def test_description_falls_back_to_content(self):
        result = normalize_memory_card({"content": "from content"})
        assert result.description == "from content"

    def test_description_preferred_over_content(self):
        result = normalize_memory_card({"description": "desc", "content": "content"})
        assert result.description == "desc"

    def test_task_description_falls_back_to_context(self):
        result = normalize_memory_card({"context": "ctx"})
        assert result.task_description == "ctx"

    def test_task_description_summary_falls_back_to_context_summary(self):
        result = normalize_memory_card({"context_summary": "s"})
        assert result.task_description_summary == "s"

    def test_explanation_non_dict_becomes_empty(self):
        result = normalize_memory_card({"explanation": "just a string"})
        assert result.explanation.explanations == []
        assert result.explanation.summary == ""

    def test_explanation_list_becomes_empty(self):
        result = normalize_memory_card({"explanation": [1, 2, 3]})
        assert result.explanation.explanations == []
        assert result.explanation.summary == ""

    def test_explanation_dict_preserved(self):
        expl = {"explanations": ["a", "b"], "summary": "sum"}
        result = normalize_memory_card({"explanation": expl})
        assert result.explanation.explanations == ["a", "b"]
        assert result.explanation.summary == "sum"

    def test_evolution_statistics_non_dict_becomes_empty(self):
        result = normalize_memory_card({"evolution_statistics": "bad"})
        assert result.evolution_statistics == {}

    def test_evolution_statistics_dict_preserved(self):
        stats = {"gen": 5, "improved": True}
        result = normalize_memory_card({"evolution_statistics": stats})
        assert result.evolution_statistics == stats

    def test_usage_non_dict_becomes_empty(self):
        result = normalize_memory_card({"usage": [1, 2]})
        assert result.usage == {}

    def test_usage_dict_preserved(self):
        usage = {"count": 3}
        result = normalize_memory_card({"usage": usage})
        assert result.usage == usage

    def test_lists_coerced_via_to_list(self):
        result = normalize_memory_card({"programs": "single"})
        assert result.programs == ["single"]

    def test_none_lists_become_empty(self):
        result = normalize_memory_card({"keywords": None})
        assert result.keywords == []

    def test_last_generation_non_int(self):
        result = normalize_memory_card({"last_generation": "abc"})
        assert result.last_generation == 0

    def test_last_generation_valid(self):
        result = normalize_memory_card({"last_generation": 42})
        assert result.last_generation == 42

    def test_strategy_preserved(self):
        result = normalize_memory_card({"strategy": "exploration"})
        assert result.strategy == "exploration"

    def test_strategy_empty_when_missing(self):
        result = normalize_memory_card({})
        assert result.strategy == ""

    def test_full_roundtrip(self):
        card = {
            "id": "test-1",
            "category": "insight",
            "description": "Use simulated annealing",
            "task_description": "Solve TSP",
            "task_description_summary": "TSP solver",
            "strategy": "exploitation",
            "last_generation": 15,
            "programs": ["p1", "p2"],
            "aliases": ["SA"],
            "keywords": ["annealing", "local-search"],
            "evolution_statistics": {"improved_count": 3},
            "explanation": {"explanations": ["tried SA"], "summary": "SA works"},
            "works_with": ["idea-2"],
            "links": ["idea-3"],
            "usage": {"times_used": 7},
        }
        result = normalize_memory_card(card)
        assert result.id == "test-1"
        assert result.category == "insight"
        assert result.description == "Use simulated annealing"
        assert result.last_generation == 15
        assert result.programs == ["p1", "p2"]
        assert result.explanation.summary == "SA works"

    def test_does_not_mutate_input(self):
        original = {"id": "x", "description": "d", "programs": ["p"]}
        copy = dict(original)
        normalize_memory_card(original)
        assert original == copy


# ===========================================================================
# normalize_memory_card — program cards
# ===========================================================================


class TestNormalizeProgramCard:
    def test_detected_by_category(self):
        result = normalize_memory_card({"category": "program"})
        assert isinstance(result, ProgramCard)
        assert result.category == "program"

    def test_detected_by_program_id(self):
        """Even without category=program, program_id triggers program path."""
        result = normalize_memory_card({"program_id": "p1"})
        assert isinstance(result, ProgramCard)
        assert result.category == "program"

    def test_exact_key_set(self):
        result = normalize_memory_card({"category": "program", "program_id": "p1"})
        assert isinstance(result, ProgramCard)

    def test_fitness_from_string(self):
        result = normalize_memory_card({"category": "program", "fitness": "3.14"})
        assert result.fitness == 3.14

    def test_fitness_none_when_missing(self):
        result = normalize_memory_card({"category": "program"})
        assert result.fitness is None

    def test_fitness_invalid_returns_none(self):
        result = normalize_memory_card({"category": "program", "fitness": "abc"})
        assert result.fitness is None

    def test_connected_ideas_preserved(self):
        ideas = [{"idea_id": "i1", "description": "d1"}]
        result = normalize_memory_card(
            {"category": "program", "connected_ideas": ideas}
        )
        assert len(result.connected_ideas) == 1
        assert result.connected_ideas[0].idea_id == "i1"
        assert result.connected_ideas[0].description == "d1"

    def test_extra_fields_stripped(self):
        result = normalize_memory_card(
            {
                "category": "program",
                "program_id": "p1",
                "links": ["l1"],
                "strategy": "hybrid",
                "keywords": ["k1"],
                "aliases": ["a1"],
            }
        )
        assert "links" not in result
        assert "strategy" not in result
        assert "keywords" not in result
        assert "aliases" not in result

    def test_code_preserved(self):
        result = normalize_memory_card({"category": "program", "code": "def f(): pass"})
        assert result.code == "def f(): pass"

    def test_code_empty_when_missing(self):
        result = normalize_memory_card({"category": "program"})
        assert result.code == ""

    def test_description_falls_back_to_content(self):
        result = normalize_memory_card({"category": "program", "content": "prog desc"})
        assert result.description == "prog desc"

    def test_task_description_falls_back_to_context(self):
        result = normalize_memory_card({"category": "program", "context": "ctx"})
        assert result.task_description == "ctx"


# ===========================================================================
# Edge cases / potential bugs
# ===========================================================================


class TestNormalizeEdgeCases:
    def test_category_with_whitespace_not_stripped(self):
        """Current behavior: category is str() of raw value, no strip."""
        result = normalize_memory_card({"category": " general "})
        # This documents actual behavior — category is NOT stripped
        assert result.category == " general "

    def test_empty_string_program_id_does_not_trigger_program_path(self):
        """program_id="" is falsy, should NOT trigger program card path."""
        result = normalize_memory_card({"program_id": ""})
        assert isinstance(result, MemoryCard)

    def test_zero_program_id_triggers_program_path(self):
        """FIXED: program_id=0 → _str_or_empty(0) → "0" → truthy → program card."""
        result = normalize_memory_card({"program_id": 0})
        assert isinstance(result, ProgramCard)
        assert result.program_id == "0"

    def test_false_program_id_triggers_program_path(self):
        """FIXED: program_id=False → _str_or_empty(False) → "False" → truthy."""
        result = normalize_memory_card({"program_id": False})
        assert isinstance(result, ProgramCard)

    def test_none_program_id_does_not_trigger(self):
        result = normalize_memory_card({"program_id": None})
        assert isinstance(result, MemoryCard)

    def test_explanation_with_extra_keys_preserved(self):
        """explanation dict may have extra keys beyond explanations/summary."""
        expl = {"explanations": ["a"], "summary": "s", "extra": "val"}
        normalize_memory_card({"explanation": expl})
        # Only explanations and summary are extracted
        assert "extra" not in MemoryCardExplanation.model_fields

    def test_aliases_with_ideas_tracker_dict_format(self):
        """Ideas tracker stores aliases as list[dict] version history, not list[str].

        RecordCardExtended.aliases appends dicts like:
            {"exp1-prog1": {"description": "old", "programs": ["p1"], "explanations": ["e"]}}
        This must pass through normalize_memory_card without crashing (Bug #2, PR #161).
        """
        aliases = [
            {
                "exp1-prog1": {
                    "description": "old description",
                    "programs": ["p1", "p2"],
                    "explanations": ["initial explanation"],
                }
            },
            {
                "exp2-prog3": {
                    "description": "updated description",
                    "programs": ["p3"],
                    "explanations": ["revised"],
                }
            },
        ]
        result = normalize_memory_card(
            {"id": "idea-1", "description": "current", "aliases": aliases}
        )
        assert isinstance(result, MemoryCard)
        assert result.aliases == aliases
        assert len(result.aliases) == 2
        assert isinstance(result.aliases[0], dict)

    def test_aliases_mixed_types(self):
        """Aliases can mix strings and dicts (legacy + ideas_tracker)."""
        aliases = ["simple-alias", {"exp1-prog1": {"description": "old"}}]
        result = normalize_memory_card({"id": "idea-2", "aliases": aliases})
        assert result.aliases == aliases

    def test_nested_dict_in_evolution_statistics(self):
        stats = {"nested": {"deep": True}}
        result = normalize_memory_card({"evolution_statistics": stats})
        assert result.evolution_statistics["nested"]["deep"] is True
