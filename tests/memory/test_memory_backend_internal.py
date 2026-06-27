"""Cycle 3: Deeper coverage for AmemGamMemory internals.

Tests apply_merges, _insert_new_card rebuild trigger,
_build_entity_meta, _concept_to_card, _ensure_card_id, and
save_card branching logic.
"""

from gigaevo.memory.shared_memory.card_conversion import (
    build_entity_meta,
    concept_to_card,
    normalize_memory_card,
)
from tests.fakes.agentic_memory import make_test_memory


def _make_memory(tmp_path, **overrides):
    return make_test_memory(tmp_path, **overrides)


# ===========================================================================
# apply_merges
# ===========================================================================


class TestApplyMerges:
    """apply_merges overwrites each target card in place and returns the ids
    that landed (the librarian's reconcile hop pre-computes the merged cards)."""

    def test_overwrites_target_and_returns_id(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "old", "programs": ["p1"]})

        merged = normalize_memory_card(
            {"id": "c1", "description": "new info", "programs": ["p1", "p2"]}
        )
        result = mem.apply_merges([("c1", merged)])
        assert result == ["c1"]

        card = mem.get_card("c1")
        assert card.description == "new info"
        assert "p2" in card.programs

    def test_applies_multiple_targets(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "card 1"})
        mem.save_card({"id": "c2", "description": "card 2"})

        merges = [
            ("c1", normalize_memory_card({"id": "c1", "description": "updated 1"})),
            ("c2", normalize_memory_card({"id": "c2", "description": "updated 2"})),
        ]
        result = mem.apply_merges(merges)
        assert set(result) == {"c1", "c2"}


# ===========================================================================
# _insert_new_card rebuild trigger
# ===========================================================================


class TestSaveCardCoreRebuild:
    def test_rebuild_called_after_interval(self, tmp_path):
        mem = _make_memory(tmp_path, rebuild_interval=3)
        calls = []
        original_rebuild = mem.rebuild
        mem.rebuild = lambda: (calls.append(1), original_rebuild())[1]
        for i in range(3):
            mem.save_card({"id": f"c{i}", "description": f"card {i}"})
        assert len(calls) == 1

    def test_no_rebuild_before_interval(self, tmp_path):
        mem = _make_memory(tmp_path, rebuild_interval=10)
        mem.save_card({"id": "c1", "description": "card"})
        assert mem._iters_after_rebuild == 1


# ===========================================================================
# _ensure_card_id
# ===========================================================================


class TestEnsureCardId:
    def test_existing_id_preserved(self, tmp_path):
        mem = _make_memory(tmp_path)
        card = normalize_memory_card({"id": "my-id", "description": "test"})
        assert mem.card_store.ensure_id(card) == "my-id"

    def test_empty_id_gets_generated(self, tmp_path):
        mem = _make_memory(tmp_path)
        card = normalize_memory_card({"id": "", "description": "test"})
        result = mem.card_store.ensure_id(card)
        assert result.startswith("mem-")
        assert card.id == result  # Mutates the card dict

    def test_whitespace_id_gets_generated(self, tmp_path):
        mem = _make_memory(tmp_path)
        card = normalize_memory_card({"id": "   ", "description": "test"})
        result = mem.card_store.ensure_id(card)
        assert result.startswith("mem-")

    def test_no_id_key_gets_generated(self, tmp_path):
        mem = _make_memory(tmp_path)
        card = normalize_memory_card({"description": "test"})
        result = mem.card_store.ensure_id(card)
        assert result.startswith("mem-")


# ===========================================================================
# _concept_to_card
# ===========================================================================


class TestConceptToCard:
    def test_basic_roundtrip(self, tmp_path):
        _make_memory(tmp_path)
        content = {
            "id": "c1",
            "category": "general",
            "description": "test idea",
            "task_description": "solve it",
            "task_description_summary": "solver",
        }
        card = concept_to_card(content, fallback_id="fb")
        assert card.id == "c1"
        assert card.description == "test idea"
        assert card.task_description == "solve it"

    def test_fallback_id_used(self, tmp_path):
        _make_memory(tmp_path)
        card = concept_to_card({}, fallback_id="fb-1")
        assert card.id == "fb-1"

    def test_program_card_concept(self, tmp_path):
        _make_memory(tmp_path)
        content = {
            "id": "p1",
            "category": "program",
            "program_id": "prog-1",
            "fitness": 90.5,
            "code": "def f(): pass",
        }
        card = concept_to_card(content, fallback_id="fb")
        assert card.category == "program"
        assert card.program_id == "prog-1"
        assert card.fitness == 90.5


# ===========================================================================
# _build_entity_meta
# ===========================================================================


class TestBuildEntityMeta:
    def test_basic(self, tmp_path):
        _make_memory(tmp_path)
        card = normalize_memory_card(
            {
                "id": "c1",
                "description": "Use simulated annealing for TSP",
                "task_description_summary": "TSP solver",
                "keywords": ["SA", "TSP"],
            }
        )
        name, tags, when_to_use = build_entity_meta(card)

        # Name derived from description (first N chars)
        assert "simulated annealing" in name.lower() or "local search" in name.lower()
        # Tags include keywords and category
        tags_lower = [t.lower() for t in tags]
        assert any("annealing" in t or "tsp" in t for t in tags_lower)
        # when_to_use references task or description content
        assert "TSP" in when_to_use or "simulated" in when_to_use.lower()

    def test_empty_card(self, tmp_path):
        _make_memory(tmp_path)
        card = normalize_memory_card({})
        name, tags, when_to_use = build_entity_meta(card)
        # Even empty card produces valid metadata
        assert isinstance(name, str)
        assert isinstance(tags, list)
        # Tags at minimum contain category
        assert any("general" in t.lower() for t in tags) or tags == []


# ===========================================================================
# save_card branching: existing ID → update path
# ===========================================================================


class TestSaveCardBranching:
    def test_existing_id_overwrites(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "v1"})
        mem.save_card({"id": "c1", "description": "v2"})
        assert len(mem.card_store.cards) == 1
        assert mem.get_card("c1").description == "v2"

    def test_new_id_adds_distinct_card(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "first"})
        mem.save_card({"id": "c2", "description": "second"})
        assert set(mem.card_store.cards) == {"c1", "c2"}

    def test_program_card_is_stored(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(
            {
                "category": "program",
                "program_id": "p1",
                "description": "prog",
                "fitness": 80.0,
            }
        )
        assert len(mem.card_store.cards) == 1
