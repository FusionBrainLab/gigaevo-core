"""Tests for ideas_tracker data components: RecordBank, RecordCardExtended,
RecordListV2, IncomingIdeas, ProgramRecord, normalize_improvement_item.

Pure data structures — no external dependencies.
"""

import pytest

from gigaevo.memory.ideas_tracker.components.data_components import (
    IncomingIdeas,
    ProgramRecord,
    RecordBank,
    RecordCardExtended,
    RecordListV2,
    normalize_improvement_item,
    normalize_improvements,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_idea_dict(**overrides):
    base = {
        "id": "idea-1",
        "category": "general",
        "description": "Use simulated annealing",
        "task_description": "Solve TSP",
        "strategy": "exploitation",
        "programs": ["prog-1"],
        "change_motivation": "Improves convergence",
    }
    base.update(overrides)
    return base


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
# ProgramRecord
# ===========================================================================


class TestProgramRecord:
    def test_defaults(self):
        p = ProgramRecord()
        assert p.id == ""
        assert p.fitness == 0.0
        assert p.generation == 0
        assert p.parents == []

    def test_to_dict(self):
        p = ProgramRecord(id="p1", fitness=85.0, generation=5)
        d = p.to_dict()
        assert d["id"] == "p1"
        assert d["fitness"] == 85.0
        assert d["generation"] == 5
        assert "code" in d

    def test_fields_independent(self):
        p1 = ProgramRecord()
        p2 = ProgramRecord()
        p1.parents.append("x")
        assert p2.parents == []


# ===========================================================================
# RecordCardExtended
# ===========================================================================


class TestRecordCardExtended:
    def test_valid_construction(self):
        card = RecordCardExtended(**_make_idea_dict())
        assert card.id == "idea-1"
        assert card.description == "Use simulated annealing"
        assert card.explanation == {
            "explanations": ["Improves convergence"],
            "summary": "",
        }

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValueError, match="Missing required fields"):
            RecordCardExtended(id="x", description="d")

    def test_update_idea_appends_program(self):
        card = RecordCardExtended(**_make_idea_dict())
        card.update_idea("exp1", "prog-2", generation=10)
        assert "prog-2" in card.programs
        assert card.last_generation == 10

    def test_update_idea_with_new_description_archives(self):
        card = RecordCardExtended(**_make_idea_dict())
        old_desc = card.description
        card.update_idea(
            "exp1", "prog-2", generation=10, new_description="Updated SA method"
        )
        assert card.description == "Updated SA method"
        assert len(card.aliases) == 1
        # Alias should contain the OLD description (archived before update)
        alias_key = list(card.aliases[0].keys())[0]
        assert card.aliases[0][alias_key]["description"] == old_desc

    def test_update_metadata(self):
        card = RecordCardExtended(**_make_idea_dict())
        card.update_metadata(
            keywords=["SA", "local-search"],
            summary="SA works",
            task_description_summary="TSP solver",
        )
        assert card.keywords == ["SA", "local-search"]
        assert card.explanation["summary"] == "SA works"
        assert card.task_description_summary == "TSP solver"

    def test_add_explanation(self):
        card = RecordCardExtended(**_make_idea_dict())
        card.add_explanation("Second explanation")
        assert "Second explanation" in card.explanation["explanations"]


# ===========================================================================
# RecordListV2
# ===========================================================================


class TestRecordListV2:
    def test_add_and_find(self):
        rl = RecordListV2(max_ideas=10)
        rl.add_idea(_make_idea_dict(id="i1"))
        assert rl.find_idea_index("i1") == 0
        assert rl.num_ideas == 1

    def test_is_full(self):
        rl = RecordListV2(max_ideas=1)
        rl.add_idea(_make_idea_dict(id="i1"))
        assert rl.is_full()

    def test_add_when_full_raises(self):
        rl = RecordListV2(max_ideas=1)
        rl.add_idea(_make_idea_dict(id="i1"))
        with pytest.raises(ValueError, match="full"):
            rl.add_idea(_make_idea_dict(id="i2"))

    def test_remove_idea(self):
        rl = RecordListV2(max_ideas=10)
        rl.add_idea(_make_idea_dict(id="i1"))
        assert rl.remove_idea("i1") is True
        assert rl.num_ideas == 0

    def test_remove_nonexistent(self):
        rl = RecordListV2(max_ideas=10)
        assert rl.remove_idea("nope") is False

    def test_get_idea(self):
        rl = RecordListV2(max_ideas=10)
        rl.add_idea(_make_idea_dict(id="i1"))
        idea = rl.get_idea("i1")
        assert idea.id == "i1"

    def test_get_idea_not_found_raises(self):
        rl = RecordListV2(max_ideas=10)
        with pytest.raises(ValueError):
            rl.get_idea("nope")

    def test_exclude_inactive_ideas(self):
        rl = RecordListV2(max_ideas=10)
        rl.add_idea(_make_idea_dict(id="i1", last_generation=1))
        rl.add_idea(_make_idea_dict(id="i2", last_generation=10))
        excluded = rl.exclude_inactive_ideas(generation=10, delta=5)
        assert len(excluded) == 1
        assert excluded[0].id == "i1"
        assert rl.num_ideas == 1  # i1 removed


# ===========================================================================
# RecordBank
# ===========================================================================


class TestRecordBank:
    def test_add_and_get(self):
        bank = RecordBank(list_max_ideas=5)
        bank.add_idea(
            "SA refinement",
            "prog-1",
            generation=5,
            category="opt",
            strategy="exploit",
            task_description="TSP",
            change_motivation="convergence",
        )
        assert len(bank.uuids) == 1
        idea = bank.get_idea(bank.uuids[0])
        assert "SA refinement" in idea.description

    def test_modify_idea(self):
        bank = RecordBank(list_max_ideas=5)
        bank.add_idea(
            "SA",
            "prog-1",
            generation=1,
            category="",
            strategy="",
            task_description="",
            change_motivation="",
        )
        idea_id = bank.uuids[0]
        bank.modify_idea(idea_id, new_programs=["prog-2"], new_generation=5)
        idea = bank.get_idea(idea_id)
        assert "prog-2" in idea.programs
        assert idea.last_generation == 5

    def test_modify_nonexistent_raises(self):
        bank = RecordBank(list_max_ideas=5)
        with pytest.raises(ValueError, match="No idea"):
            bank.modify_idea("nope", new_programs=["p"])

    def test_remove_idea(self):
        bank = RecordBank(list_max_ideas=5)
        bank.add_idea(
            "SA",
            "prog-1",
            generation=1,
            category="",
            strategy="",
            task_description="",
            change_motivation="",
        )
        idea_id = bank.uuids[0]
        bank.remove_idea(idea_id)
        assert idea_id not in bank.uuids

    def test_remove_nonexistent_raises(self):
        bank = RecordBank(list_max_ideas=5)
        with pytest.raises(ValueError):
            bank.remove_idea("nope")

    def test_get_inactive_ideas(self):
        bank = RecordBank(list_max_ideas=10)
        bank.add_idea(
            "old",
            "p1",
            generation=1,
            category="",
            strategy="",
            task_description="",
            change_motivation="",
        )
        bank.add_idea(
            "new",
            "p2",
            generation=10,
            category="",
            strategy="",
            task_description="",
            change_motivation="",
        )
        inactive = bank.get_inactive_ideas(generation=10, delta=5)
        assert len(inactive) == 1
        assert inactive[0].description == "old"

    def test_all_ideas_cards(self):
        bank = RecordBank(list_max_ideas=5)
        for i in range(3):
            bank.add_idea(
                f"idea-{i}",
                f"p-{i}",
                generation=i,
                category="",
                strategy="",
                task_description="",
                change_motivation="",
            )
        all_cards = bank.all_ideas_cards()
        assert len(all_cards) == 3

    def test_import_idea_forced_deduplicates_id(self):
        """import_idea_extended(is_forced=True) avoids asdict() crash."""
        bank = RecordBank(list_max_ideas=5)
        bank.add_idea(
            "first",
            "p1",
            generation=1,
            category="",
            strategy="",
            task_description="",
            change_motivation="",
        )
        existing_id = bank.uuids[0]

        card = RecordCardExtended(**_make_idea_dict(id=existing_id))
        bank.import_idea_extended(card, is_forced=True)
        # Should have assigned a new ID since existing_id was taken
        assert len(bank.uuids) == 2
        assert bank.uuids[1] != existing_id

    def test_import_idea_non_forced_crashes_on_asdict(self):
        """BUG: RecordCardExtended custom __init__ doesn't initialize
        all dataclass fields (keywords, evolution_statistics, works_with,
        links, usage), so asdict() raises AttributeError when is_forced=False.

        The custom __init__ uses setattr only for provided kwargs,
        but dataclass.asdict() expects ALL declared fields to exist.
        """
        bank = RecordBank(list_max_ideas=5)
        card = RecordCardExtended(**_make_idea_dict())
        # Verify the missing attributes directly
        for attr in (
            "keywords",
            "evolution_statistics",
            "works_with",
            "links",
            "usage",
        ):
            assert not hasattr(card, attr), f"Expected {attr} to be missing"
        with pytest.raises(AttributeError):
            bank.import_idea_extended(card, is_forced=False)


# ===========================================================================
# IncomingIdeas
# ===========================================================================


class TestIncomingIdeas:
    def test_construction(self):
        ideas = IncomingIdeas(
            [
                {"description": "idea A", "explanation": "reason A"},
                {"description": "idea B", "explanation": "reason B"},
            ]
        )
        assert len(ideas.ideas) == 2
        assert ideas.new_ideas_count == 2
        assert ideas.present_ideas_count == 0

    def test_update_marks_classified(self):
        ideas = IncomingIdeas(
            [
                {"description": "idea A"},
                {"description": "idea B"},
            ]
        )
        ideas.update_idea(1, target_idea_id="uuid-1", rewrite=False)
        assert ideas.new_ideas_count == 1
        assert ideas.present_ideas_count == 1

    def test_get_list_of_ideas(self):
        ideas = IncomingIdeas(
            [
                {"description": "idea A"},
                {"description": "idea B"},
            ]
        )
        text = ideas.get_list_of_ideas()
        assert "1) idea A" in text
        assert "2) idea B" in text

    def test_classified_excluded_from_list(self):
        ideas = IncomingIdeas(
            [
                {"description": "idea A"},
                {"description": "idea B"},
            ]
        )
        ideas.update_idea(1, target_idea_id="uuid-1", rewrite=False)
        ideas.update_mapping()
        text = ideas.get_list_of_ideas()
        assert "idea A" not in text
        assert "1) idea B" in text

    def test_empty_input(self):
        ideas = IncomingIdeas([])
        assert ideas.new_ideas_count == 0
        assert ideas.get_list_of_ideas() == ""

    def test_string_ideas_normalized(self):
        ideas = IncomingIdeas(["just a string idea"])
        assert len(ideas.ideas) == 1
        assert ideas.ideas[0]["description"] == "just a string idea"
