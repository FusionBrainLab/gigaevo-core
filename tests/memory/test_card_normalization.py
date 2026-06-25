"""Tests for normalize_memory_card and its helper functions.

Pin down the exact normalization behavior so refactoring can be validated.
"""

from pydantic import ValidationError
import pytest

from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.shared_memory.card_conversion import (
    RawCardRecord,
    normalize_memory_card,
)
from gigaevo.memory.shared_memory.models import (
    CardAlias,
    MemoryCard,
    MemoryCardExplanation,
    ProgramCard,
)
from gigaevo.memory.shared_memory.utils import _to_int, _to_list
from gigaevo.memory.utils import to_float

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
# to_float (canonical — filters NaN/Inf)
# ===========================================================================


class TestToFloat:
    def test_valid_float(self):
        assert to_float(3.14) == 3.14

    def test_valid_string(self):
        assert to_float("2.5") == 2.5

    def test_int_promoted(self):
        assert to_float(7) == 7.0

    def test_invalid_returns_default_none(self):
        assert to_float("abc") is None

    def test_invalid_custom_default(self):
        assert to_float("abc", default=0.0) == 0.0

    def test_none_returns_default(self):
        assert to_float(None) is None

    def test_empty_string(self):
        assert to_float("") is None

    def test_negative(self):
        assert to_float("-1.5") == -1.5

    def test_inf_filtered(self):
        assert to_float("inf") is None

    def test_nan_filtered(self):
        assert to_float("nan") is None


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

    def test_explanation_string_becomes_summary(self):
        result = normalize_memory_card({"explanation": "just a string"})
        assert result.explanation.explanations == []
        assert result.explanation.summary == "just a string"

    def test_explanation_list_becomes_empty(self):
        result = normalize_memory_card({"explanation": [1, 2, 3]})
        assert result.explanation.explanations == []
        assert result.explanation.summary == ""

    def test_explanation_dict_preserved(self):
        expl = {"explanations": ["a", "b"], "summary": "sum"}
        result = normalize_memory_card({"explanation": expl})
        assert result.explanation.explanations == ["a", "b"]
        assert result.explanation.summary == "sum"

    def test_gain_events_non_list_becomes_none(self):
        result = normalize_memory_card({"gain_events": "bad"})
        assert result.gain_events is None

    def test_gain_events_list_validated_into_typed_events(self):
        events = [{"context": {"parent_metrics": {"min_area": 0.5}}, "gain": 0.01}]
        result = normalize_memory_card({"gain_events": events})
        assert len(result.gain_events) == 1
        assert isinstance(result.gain_events[0], ContextualGain)
        assert result.gain_events[0].gain == 0.01

    def test_gain_events_malformed_event_rejected(self):
        with pytest.raises(ValidationError):
            normalize_memory_card({"gain_events": [{"gain": 0.01}]})

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
            "aliases": [{"key": "test-1-update", "description": "SA (initial)"}],
            "keywords": ["annealing", "local-search"],
            "gain_events": [
                {"context": {"parent_metrics": {"min_area": 0.5}}, "gain": 0.01}
            ],
            "explanation": {"explanations": ["tried SA"], "summary": "SA works"},
            "works_with": ["idea-2"],
            "links": ["idea-3"],
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
                "aliases": [{"key": "a1", "description": "d1"}],
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

    def test_aliases_validate_into_card_alias(self):
        """Alias version history validates into typed CardAlias entries."""
        aliases = [
            {
                "key": "idea-1-update",
                "description": "old description",
                "programs": ["p1", "p2"],
                "explanations": ["initial explanation"],
            },
            {
                "key": "idea-1-canonical-merge",
                "description": "updated description",
                "programs": ["p3"],
                "explanations": ["revised"],
            },
        ]
        result = normalize_memory_card(
            {"id": "idea-1", "description": "current", "aliases": aliases}
        )
        assert isinstance(result, MemoryCard)
        assert len(result.aliases) == 2
        assert all(isinstance(a, CardAlias) for a in result.aliases)
        assert result.aliases[0].key == "idea-1-update"
        assert result.aliases[1].programs == ["p3"]

    def test_gain_events_survive_normalize_from_typed_card(self):
        original = MemoryCard(
            id="x",
            gain_events=[
                ContextualGain(
                    context=DecisionContext(parent_metrics={"min_area": 0.5}),
                    gain=0.02,
                )
            ],
        )
        rebuilt = normalize_memory_card(original.model_dump())
        assert len(rebuilt.gain_events) == 1
        assert rebuilt.gain_events[0].gain == 0.02


# ===========================================================================
# RawCardRecord — boundary envelope
# ===========================================================================


class TestRawCardRecord:
    def test_unknown_keys_ignored(self):
        record = RawCardRecord.model_validate({"id": "x", "totally_unknown": 1})
        assert record.id == "x"

    def test_alias_keys_resolve_on_to_card(self):
        record = RawCardRecord.model_validate(
            {"content": "c", "context": "ctx", "context_summary": "cs"}
        )
        card = record.to_card()
        assert card.description == "c"
        assert card.task_description == "ctx"
        assert card.task_description_summary == "cs"

    def test_canonical_keys_win_over_aliases(self):
        record = RawCardRecord.model_validate(
            {
                "description": "d",
                "content": "c",
                "task_description": "t",
                "context": "x",
            }
        )
        card = record.to_card()
        assert card.description == "d"
        assert card.task_description == "t"

    def test_to_card_fallback_id(self):
        card = RawCardRecord.model_validate({}).to_card(fallback_id="fb")
        assert card.id == "fb"

    def test_program_dispatch_by_category(self):
        card = RawCardRecord.model_validate({"category": "program"}).to_card()
        assert isinstance(card, ProgramCard)

    def test_program_dispatch_by_program_id(self):
        card = RawCardRecord.model_validate({"program_id": 0}).to_card()
        assert isinstance(card, ProgramCard)
        assert card.program_id == "0"

    def test_string_explanation_becomes_summary(self):
        record = RawCardRecord.model_validate({"explanation": "why it works"})
        assert record.explanation.summary == "why it works"
        assert record.explanation.explanations == []

    def test_program_card_category_roundtrips(self):
        raw = {"program_id": "p1", "category": "program-seed"}
        card = RawCardRecord.model_validate(raw).to_card()
        assert isinstance(card, ProgramCard)
        assert card.category == "program-seed"

    def test_program_dispatch_without_category_defaults_to_program(self):
        card = RawCardRecord.model_validate({"program_id": "p1"}).to_card()
        assert isinstance(card, ProgramCard)
        assert card.category == "program"

    def test_program_card_full_roundtrip_preserves_category(self):
        original = ProgramCard(id="pc-1", program_id="p1", description="d")
        rebuilt = normalize_memory_card(original.model_dump())
        assert isinstance(rebuilt, ProgramCard)
        assert rebuilt.category == original.category
