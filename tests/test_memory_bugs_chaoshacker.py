"""Tests exposing bugs found by chaos-hacker adversarial review.

Each test documents a real bug. Tests that expose currently-broken behavior
are marked with comments explaining the issue. When the bug is fixed,
the test assertion should be updated to reflect correct behavior.
"""

import json
import re
import uuid
from unittest.mock import MagicMock, patch

import pytest

from gigaevo.memory.shared_memory.memory import AmemGamMemory, normalize_memory_card
from gigaevo.memory.shared_memory.card_update_dedup import (
    _extract_json_object,
    append_unique_text,
)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


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
# BUG 1 (CRITICAL): Corrupt api_index.json → silent empty start
# ===========================================================================


class TestBug1CorruptIndexFile:
    def test_partial_json_silently_starts_empty(self, tmp_path):
        """If api_index.json contains truncated JSON (from a crash mid-write),
        AmemGamMemory silently starts with empty state instead of raising.

        This documents current behavior. A fix would use atomic writes.
        """
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir(parents=True)
        index_file = mem_dir / "api_index.json"

        # Simulate crash mid-write: valid JSON start, truncated
        index_file.write_text('{"memory_cards": {"c1": {"id": "c1", "descr')

        mem = _make_memory(tmp_path)
        # BUG: silently lost all data
        assert mem.memory_cards == {}

    def test_valid_index_loads_correctly(self, tmp_path):
        """Contrast: valid JSON loads fine."""
        mem1 = _make_memory(tmp_path)
        mem1.save_card({"id": "c1", "description": "test"})

        mem2 = _make_memory(tmp_path)
        assert mem2.get_card("c1") is not None


# ===========================================================================
# BUG 5 (HIGH): Substring search matches too broadly
# ===========================================================================


class TestBug5SubstringSearch:
    def test_short_token_matches_too_broadly(self, tmp_path):
        """Token 'a' matches any card containing 'a' anywhere in ANY field.

        BUG: _search_local_cards uses `tok in haystack` (substring match)
        across description + task_description_summary + task_description +
        keywords + category. The category field defaults to "general" which
        contains 'a', so single-char token 'a' matches EVERY card.
        """
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "xyz specific topic",
                        "task_description": "", "task_description_summary": ""})

        result = mem.search("a")
        # BUG: "a" matches because category="general" contains "a"
        # This means searching for "a" returns ALL cards regardless of content
        assert "c1" in result  # Documents the bug: false positive

    def test_single_char_token_overmatch(self, tmp_path):
        """Single-char tokens match almost everything."""
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "database management"})
        mem.save_card({"id": "c2", "description": "python programming"})

        result = mem.search("a")
        # "a" is in "database" and "management" → c1 matches
        assert "c1" in result
        # BUG: "a" is NOT in "python" but IS in "programming" → c2 also matches
        # This is overly broad — a search for "a" shouldn't match "programming"
        # Documenting actual behavior:
        assert "c2" in result  # This is the bug: "a" in "programming"


# ===========================================================================
# BUG 6 (MEDIUM): ID collision with 48-bit entropy
# ===========================================================================


class TestBug6IDCollision:
    def test_collision_silently_overwrites(self, tmp_path):
        """If uuid4 generates the same 12-hex prefix twice, the first card
        is silently overwritten. No collision detection.
        """
        mem = _make_memory(tmp_path)

        # Mock uuid4 to return same value twice
        fixed_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        with patch("gigaevo.memory.shared_memory.memory.uuid.uuid4", return_value=fixed_uuid):
            id1 = mem.save_card({"description": "first card"})
            id2 = mem.save_card({"description": "second card"})

        # Both got the same auto-generated ID
        assert id1 == id2
        # BUG: first card silently overwritten
        assert mem.get_card(id1)["description"] == "second card"
        assert len(mem.memory_cards) == 1  # Only one card exists


# ===========================================================================
# BUG 8 (MEDIUM): Greedy regex grabs wrong braces
# ===========================================================================


class TestBug8GreedyRegex:
    def test_reasoning_with_braces_before_json(self):
        """LLM reasoning contains literal braces before the actual JSON.
        Greedy .* captures from first { to last }, yielding invalid JSON.
        """
        text = 'I considered {various factors}. My decision: {"action": "discard", "duplicate_of": "c1"}'
        result = _extract_json_object(text)
        # BUG: regex captures '{various factors}...{"action":...'
        # json.loads fails on this, so result should be None or fall through
        # Actually, the greedy regex captures from first { to last }:
        # '{various factors}. My decision: {"action": "discard", "duplicate_of": "c1"}'
        # This is invalid JSON, so json.loads fails → returns None
        # Documenting actual behavior:
        assert result is None  # The correct JSON is lost

    def test_clean_json_in_prose_works(self):
        """When JSON has no preceding braces, extraction works fine."""
        text = 'My decision is: {"action": "add"}'
        result = _extract_json_object(text)
        assert result == {"action": "add"}

    def test_nested_braces_in_json_works(self):
        """Nested braces within the actual JSON are handled."""
        text = '{"action": "update", "meta": {"key": "val"}}'
        result = _extract_json_object(text)
        assert result["action"] == "update"


# ===========================================================================
# BUG 11 (MEDIUM): O(n^2) persist — full JSON on every save
# ===========================================================================


class TestBug11PersistScaling:
    def test_index_file_grows_with_card_count(self, tmp_path):
        """Each save_card serializes the ENTIRE memory_cards dict.
        Verify index file size grows linearly with card count.
        """
        mem = _make_memory(tmp_path)
        sizes = []
        for i in range(20):
            mem.save_card({"id": f"c{i}", "description": f"card {i}" * 10})
            size = mem.index_file.stat().st_size
            sizes.append(size)

        # Index file should grow ~linearly with card count
        assert sizes[-1] > sizes[0] * 5  # At least 5x growth from 1→20 cards
        # This documents the O(n) per-write behavior (total O(n^2) for n saves)


# ===========================================================================
# BUG 12 (MEDIUM): append_unique_text drops short text
# ===========================================================================


class TestBug12AppendUniqueTextSubstring:
    def test_short_text_is_substring_of_long(self):
        """'retrieval' is a substring of 'deep retrieval pipeline' →
        silently discarded even though it could be a separate concept.
        """
        result = append_unique_text(
            "deep retrieval pipeline for multi-hop verification",
            "retrieval",
        )
        # BUG: "retrieval" is dropped because it's a substring of existing text
        assert result == "deep retrieval pipeline for multi-hop verification"
        # The new text "retrieval" is silently lost

    def test_unrelated_short_text_appended(self):
        """Short text that isn't a substring gets appended correctly."""
        result = append_unique_text("deep retrieval pipeline", "crossover")
        assert "crossover" in result

    def test_exact_duplicate_correctly_dropped(self):
        """Exact duplicates should be dropped (correct behavior)."""
        result = append_unique_text("same text", "same text")
        assert result == "same text"


# ===========================================================================
# BUG (documented): program_id=0 silently lost
# ===========================================================================


class TestBugFalsyProgramId:
    def test_zero_program_id_lost(self):
        """program_id=0 → str(0 or '') → '' → falsy → general card.

        This means numeric program IDs that happen to be 0 are silently
        treated as general cards instead of program cards.
        """
        card = normalize_memory_card({"program_id": 0, "description": "prog"})
        # BUG: 0 is a valid program_id but gets lost
        assert card.get("program_id", "") == ""  # Lost!
        assert "fitness" not in card  # Not treated as program card

    def test_nonzero_numeric_program_id_works(self):
        """program_id=42 → str(42 or '')='42' → truthy → program card."""
        card = normalize_memory_card({"program_id": 42, "description": "prog"})
        assert card["category"] == "program"
        assert card["program_id"] == "42"


# ===========================================================================
# BUG 7 (MEDIUM): Update action falls through to add
# ===========================================================================


class TestBug7UpdateFallthrough:
    def test_update_target_deleted_between_score_and_apply(self, tmp_path):
        """If LLM says 'update card X' but card X no longer exists,
        the update returns empty and falls through to add.
        """
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card({"id": "existing", "description": "original"})

        # Set up LLM mock returning update action
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps({
                "action": "update",
                "updates": [{
                    "card_id": "existing",
                    "update_explanation": True,
                    "explanation_append": "new info",
                }],
            }),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm
        mem._score_retrieved_candidates = MagicMock(
            return_value=[{"card_id": "existing", "score": 0.8}]
        )

        # Delete the target card BEFORE the dedup processes
        # (simulating concurrent deletion)
        del mem.memory_cards["existing"]

        # Now save a new card — dedup will try to update "existing" but it's gone
        card_id = mem.save_card({"description": "should be deduped"})
        # BUG: Falls through to add because _apply_update_actions returns []
        stats = mem.get_card_write_stats()
        assert stats["added"] >= 2  # Both cards added despite dedup identifying duplicate


# ===========================================================================
# get_card returns mutable reference (not a bug per se, but a footgun)
# ===========================================================================


class TestGetCardMutableReference:
    def test_mutation_visible_through_reference(self, tmp_path):
        """get_card returns direct dict reference — external mutations
        are visible and will be persisted on next _persist_index call.
        """
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "original"})

        card = mem.get_card("c1")
        card["description"] = "externally mutated"
        card["injected_field"] = "sneaky"

        # Mutation is visible
        assert mem.get_card("c1")["description"] == "externally mutated"
        assert mem.get_card("c1")["injected_field"] == "sneaky"

        # And will be persisted!
        mem._persist_index()
        data = json.loads(mem.index_file.read_text())
        assert data["memory_cards"]["c1"]["injected_field"] == "sneaky"
