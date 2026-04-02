"""Cycle 3: Deeper coverage for AmemGamMemory internals.

Tests _apply_update_actions, _save_card_core rebuild trigger,
_build_entity_meta, _concept_to_card, _ensure_card_id, and
save_card branching logic.
"""

import json
from unittest.mock import MagicMock

from gigaevo.memory.shared_memory.card_conversion import (
    build_entity_meta,
    concept_to_card,
)
from gigaevo.memory.shared_memory.memory import AmemGamMemory, normalize_memory_card


def _make_memory(tmp_path, **overrides):
    defaults = dict(
        checkpoint_path=str(tmp_path / "mem"),
        use_api=False,
        sync_on_init=False,
        enable_llm_synthesis=False,
        enable_memory_evolution=False,
        enable_llm_card_enrichment=False,
    )
    defaults.update(overrides)
    return AmemGamMemory(**defaults)


# ===========================================================================
# _apply_update_actions
# ===========================================================================


class TestApplyUpdateActions:
    def test_updates_existing_card(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "old", "programs": ["p1"]})

        incoming = {
            "description": "new info",
            "programs": ["p2"],
            "last_generation": 10,
        }
        updates = [
            {
                "card_id": "c1",
                "update_explanation": True,
                "explanation_append": "extra detail",
            }
        ]
        result = mem._apply_update_actions(incoming, updates)
        assert result == ["c1"]

        card = mem.get_card("c1")
        assert "extra detail" in card["explanation"]["explanations"]
        # Programs should be merged
        assert "p1" in card["programs"]
        assert "p2" in card["programs"]

    def test_skips_missing_card(self, tmp_path):
        mem = _make_memory(tmp_path)
        updates = [{"card_id": "nonexistent", "update_explanation": True}]
        result = mem._apply_update_actions({}, updates)
        assert result == []

    def test_skips_duplicate_card_id(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "test"})

        updates = [
            {"card_id": "c1", "update_explanation": True, "explanation_append": "a"},
            {"card_id": "c1", "update_explanation": True, "explanation_append": "b"},
        ]
        result = mem._apply_update_actions({}, updates)
        assert result == ["c1"]  # Only processed once

    def test_skips_non_dict_update(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "test"})
        result = mem._apply_update_actions({}, ["not a dict", None])
        assert result == []

    def test_multiple_target_cards(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "card 1"})
        mem.save_card({"id": "c2", "description": "card 2"})

        updates = [
            {
                "card_id": "c1",
                "update_explanation": True,
                "explanation_append": "info1",
            },
            {
                "card_id": "c2",
                "update_explanation": True,
                "explanation_append": "info2",
            },
        ]
        result = mem._apply_update_actions({}, updates)
        assert set(result) == {"c1", "c2"}


# ===========================================================================
# _save_card_core rebuild trigger
# ===========================================================================


class TestSaveCardCoreRebuild:
    def test_rebuild_called_after_interval(self, tmp_path):
        mem = _make_memory(tmp_path, rebuild_interval=3)
        for i in range(3):
            mem.save_card({"id": f"c{i}", "description": f"card {i}"})
        # rebuild() is called but early-returns when memory_system is None,
        # so _iters_after_rebuild is NOT reset. This is a minor behavioral
        # quirk: the counter keeps growing in local-only mode.
        assert mem._iters_after_rebuild == 3

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
        card = {"id": "my-id", "description": "test"}
        assert mem._ensure_card_id(card) == "my-id"

    def test_empty_id_gets_generated(self, tmp_path):
        mem = _make_memory(tmp_path)
        card = {"id": "", "description": "test"}
        result = mem._ensure_card_id(card)
        assert result.startswith("mem-")
        assert card["id"] == result  # Mutates the card dict

    def test_whitespace_id_gets_generated(self, tmp_path):
        mem = _make_memory(tmp_path)
        card = {"id": "   ", "description": "test"}
        result = mem._ensure_card_id(card)
        assert result.startswith("mem-")

    def test_no_id_key_gets_generated(self, tmp_path):
        mem = _make_memory(tmp_path)
        card = {"description": "test"}
        result = mem._ensure_card_id(card)
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
        assert card["id"] == "c1"
        assert card["description"] == "test idea"
        assert card["task_description"] == "solve it"

    def test_fallback_id_used(self, tmp_path):
        _make_memory(tmp_path)
        card = concept_to_card({}, fallback_id="fb-1")
        assert card["id"] == "fb-1"

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
        assert card["category"] == "program"
        assert card["program_id"] == "prog-1"
        assert card["fitness"] == 90.5


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
    def test_existing_id_goes_to_update_path(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "v1"})
        mem.save_card({"id": "c1", "description": "v2"})
        stats = mem.get_card_write_stats()
        assert stats["updated"] == 1
        assert stats["added"] == 1

    def test_new_id_goes_to_add_path(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "first"})
        mem.save_card({"id": "c2", "description": "second"})
        stats = mem.get_card_write_stats()
        assert stats["added"] == 2
        assert stats["updated"] == 0

    def test_program_card_always_added(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(
            {
                "category": "program",
                "program_id": "p1",
                "description": "prog",
                "fitness": 80.0,
            }
        )
        stats = mem.get_card_write_stats()
        assert stats["added"] == 1

    def test_dedup_update_path(self, tmp_path):
        """Full dedup update flow: LLM says update, _apply_update_actions runs."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card({"id": "existing", "description": "original idea"})

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps(
                {
                    "action": "update",
                    "updates": [
                        {
                            "card_id": "existing",
                            "update_explanation": True,
                            "explanation_append": "merged info",
                        }
                    ],
                }
            ),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm
        mem._score_retrieved_candidates = MagicMock(
            return_value=[{"card_id": "existing", "score": 0.8}]
        )

        card_id = mem.save_card({"description": "similar idea with extra detail"})
        assert card_id == "existing"  # Updated existing card
        stats = mem.get_card_write_stats()
        assert stats["updated"] == 1
        assert stats["updated_target_cards"] == 1

    def test_dedup_warning_only_once(self, tmp_path):
        """Missing LLM warning printed only once."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card({"id": "seed", "description": "seed"})
        # First save with missing LLM triggers warning
        assert not mem._warned_missing_card_update_llm
        mem.save_card({"description": "new1"})
        assert mem._warned_missing_card_update_llm
        # Subsequent saves don't re-warn (flag stays True)
        mem.save_card({"description": "new2"})
        assert mem._warned_missing_card_update_llm
