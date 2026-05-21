"""Edge-case tests pinning observable behavior of the memory module.

Each test asserts the current contract: persistence on corruption,
ID-collision handling, JSON extraction quirks, write-amplification
shape, dedup substring semantics, and program-card normalization.
Tests serve as guard rails for changes to that contract.
"""

import json
from unittest.mock import MagicMock, patch
import uuid

from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.card_update_dedup import (
    _extract_json_object,
    append_unique_text,
)
from gigaevo.memory.shared_memory.models import ProgramCard
from tests.fakes.agentic_memory import make_test_memory

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_memory(tmp_path, **overrides):
    return make_test_memory(tmp_path, **overrides)


# ===========================================================================
# Corrupt api_index.json on load
# ===========================================================================


class TestCorruptIndexFile:
    def test_partial_json_silently_starts_empty(self, tmp_path):
        """Truncated api_index.json yields an empty in-memory state on load."""
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir(parents=True)
        index_file = mem_dir / "api_index.json"

        # Simulate crash mid-write: valid JSON start, truncated
        index_file.write_text('{"memory_cards": {"c1": {"id": "c1", "descr')

        mem = _make_memory(tmp_path)
        assert mem.card_store.cards == {}

    def test_valid_index_loads_correctly(self, tmp_path):
        """Cards saved into one instance round-trip through a fresh load."""
        mem1 = _make_memory(tmp_path)
        mem1.save_card({"id": "c1", "description": "test"})

        mem2 = _make_memory(tmp_path)
        assert mem2.get_card("c1") is not None


# ===========================================================================
# Substring search semantics
# ===========================================================================


class TestSubstringSearch:
    """Search uses word-boundary token matching."""

    def test_short_token_no_longer_matches_inside_words(self, tmp_path):
        """Token 'a' does not match inside 'general' because 'a' is not a
        standalone word there."""
        mem = _make_memory(tmp_path)
        mem.save_card(
            {
                "id": "c1",
                "description": "xyz specific topic",
                "task_description": "",
                "task_description_summary": "",
            }
        )

        result = mem.search("a")
        assert "No relevant memories found" in result

    def test_single_char_token_no_overmatch(self, tmp_path):
        """Single-char token 'a' doesn't match inside 'database' or 'programming'."""
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "database management"})
        mem.save_card({"id": "c2", "description": "python programming"})

        result = mem.search("a")
        assert "No relevant memories found" in result

    def test_whole_word_matching_still_works(self, tmp_path):
        """Whole word tokens match correctly."""
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "database management system"})
        result = mem.search("database")
        assert "c1" in result


# ===========================================================================
# Auto-generated ID collisions
# ===========================================================================


class TestIDCollision:
    def test_collision_silently_overwrites(self, tmp_path):
        """When uuid4 yields the same 12-hex prefix twice, the first card is
        overwritten without collision detection."""
        mem = _make_memory(tmp_path)

        fixed_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        with patch(
            "gigaevo.memory.shared_memory.card_store.uuid.uuid4",
            return_value=fixed_uuid,
        ):
            id1 = mem.save_card({"description": "first card"})
            id2 = mem.save_card({"description": "second card"})

        assert id1 == id2
        assert mem.get_card(id1).description == "second card"
        assert len(mem.card_store.cards) == 1


# ===========================================================================
# JSON-object extraction from LLM prose
# ===========================================================================


class TestExtractJSONObject:
    def test_reasoning_with_braces_before_json(self):
        """Brace-bearing prose ahead of the JSON yields no extraction."""
        text = 'I considered {various factors}. My decision: {"action": "discard", "duplicate_of": "c1"}'
        result = _extract_json_object(text)
        # Greedy '{...}' span fails json.loads -> None.
        assert result is None

    def test_clean_json_in_prose_works(self):
        """JSON with no preceding braces extracts cleanly."""
        text = 'My decision is: {"action": "add"}'
        result = _extract_json_object(text)
        assert result == {"action": "add"}

    def test_nested_braces_in_json_works(self):
        """Nested braces within the JSON object extract."""
        text = '{"action": "update", "meta": {"key": "val"}}'
        result = _extract_json_object(text)
        assert result["action"] == "update"


# ===========================================================================
# Index-file persist scaling
# ===========================================================================


class TestPersistScaling:
    def test_index_file_grows_with_card_count(self, tmp_path):
        """Index file grows roughly linearly as cards are appended."""
        mem = _make_memory(tmp_path)
        sizes = []
        for i in range(20):
            mem.save_card({"id": f"c{i}", "description": f"card {i}" * 10})
            size = mem.config.index_file.stat().st_size
            sizes.append(size)

        assert sizes[-1] > sizes[0] * 5  # 1->20 cards produces >=5x growth


# ===========================================================================
# append_unique_text substring semantics
# ===========================================================================


class TestAppendUniqueTextSubstring:
    def test_short_text_is_substring_of_long(self):
        """Short text that is a substring of existing text is dropped."""
        result = append_unique_text(
            "deep retrieval pipeline for multi-hop verification",
            "retrieval",
        )
        assert result == "deep retrieval pipeline for multi-hop verification"

    def test_unrelated_short_text_appended(self):
        """Short text that isn't a substring gets appended correctly."""
        result = append_unique_text("deep retrieval pipeline", "crossover")
        assert "crossover" in result

    def test_exact_duplicate_correctly_dropped(self):
        """Exact duplicates should be dropped (correct behavior)."""
        result = append_unique_text("same text", "same text")
        assert result == "same text"


# ===========================================================================
# Falsy program_id normalization
# ===========================================================================


class TestFalsyProgramId:
    """program_id values pass through string-coercion before category check."""

    def test_zero_program_id_preserved(self):
        """program_id=0 string-coerces to '0', producing a program card."""
        card = normalize_memory_card({"program_id": 0, "description": "prog"})
        assert isinstance(card, ProgramCard)
        assert card.program_id == "0"
        assert card.category == "program"

    def test_nonzero_numeric_program_id_works(self):
        card = normalize_memory_card({"program_id": 42, "description": "prog"})
        assert card.category == "program"
        assert card.program_id == "42"

    def test_none_program_id_still_general(self):
        card = normalize_memory_card({"program_id": None, "description": "d"})
        assert card.category == "general"

    def test_false_program_id_preserved(self):
        """program_id=False string-coerces to 'False', producing a program card."""
        card = normalize_memory_card({"program_id": False, "description": "d"})
        assert card.category == "program"
        assert card.program_id == "False"


# ===========================================================================
# Update action with missing target card
# ===========================================================================


class TestUpdateMissingTarget:
    def test_update_target_deleted_between_score_and_apply(self, tmp_path):
        """When an update names a card that no longer exists, the save falls
        through to add and both cards land in the store."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card({"id": "existing", "description": "original"})

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps(
                {
                    "action": "update",
                    "updates": [
                        {
                            "card_id": "existing",
                            "update_explanation": True,
                            "explanation_append": "new info",
                        }
                    ],
                }
            ),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm
        mem.dedup.score_candidates = MagicMock(
            return_value=[{"card_id": "existing", "score": 0.8}]
        )

        # Delete the target card before dedup runs (concurrent deletion).
        del mem.card_store.cards["existing"]

        mem.save_card({"description": "should be deduped"})
        stats = mem.get_card_write_stats()
        assert stats["added"] >= 2


# ===========================================================================
# get_card returns a live Pydantic model
# ===========================================================================


class TestGetCardReturnsPydanticModel:
    def test_get_card_returns_model(self, tmp_path):
        """get_card returns a Pydantic model with typed fields."""
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "original"})

        card = mem.get_card("c1")
        assert card.description == "original"
        assert card.id == "c1"
        assert card.category == "general"

    def test_model_mutation_via_validate_assignment(self, tmp_path):
        """Pydantic models with validate_assignment=True allow field mutation."""
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "original"})

        card = mem.get_card("c1")
        card.description = "mutated"
        assert mem.get_card("c1").description == "mutated"
