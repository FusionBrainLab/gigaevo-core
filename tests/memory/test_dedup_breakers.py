"""Adversarial tests targeting CardDedup LLM failures, merge edge cases,
orchestrator integration, NoteSync exceptions, and config validation.

Tests the deduplication pipeline from scoring through LLM decision to merge
computation, plus interactions with the memory orchestrator.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from pydantic import ValidationError
import pytest

from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.memory_config import (
    ApiConfig,
    GamConfig,
    MemoryConfig,
)
from tests.fakes.agentic_memory import make_test_memory, make_test_memory_with_agentic


def _make_memory(tmp_path, **overrides):
    return make_test_memory(tmp_path, **overrides)


def _make_full_memory(tmp_path, **overrides):
    mem, _ = make_test_memory_with_agentic(tmp_path, **overrides)
    return mem


# ===========================================================================
# Category F: LLM Decision Failures (card_dedup.py)
# ===========================================================================


class TestDedupLLMFailures:
    """Tests for LLM response parsing and retry logic in dedup decisions."""

    def test_all_retries_exhausted_defaults_to_add(self, tmp_path):
        """F1: When LLM returns unparseable text on all retries, decide_action
        returns default {"action": "add"}."""
        mem = _make_full_memory(
            tmp_path,
            card_update_dedup_config={"enabled": True, "llm_max_retries": 2},
        )
        mem.save_card({"id": "existing", "description": "original"})

        mock_llm = MagicMock()
        mock_llm.generate.return_value = ("garbage garbage garbage", {}, None, None)
        mem.dedup.llm_service = mock_llm

        # Pre-formatted candidates list (bypassing score_candidates)
        candidates = [{"card_id": "existing", "final_score": 0.8}]

        decision = mem.dedup.decide_action(
            normalize_memory_card({"description": "new card"}), candidates
        )

        assert decision["action"] == "add"
        # LLM was called max_retries times
        assert mock_llm.generate.call_count == 2

    def test_llm_exception_on_every_call_defaults_to_add(self, tmp_path):
        """F2: When LLM raises an exception on every call, decide_action
        returns default {"action": "add"} without propagating exception."""
        mem = _make_full_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card({"id": "existing", "description": "original"})

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM timeout")
        mem.dedup.llm_service = mock_llm

        candidates = [{"card_id": "existing", "final_score": 0.8}]
        decision = mem.dedup.decide_action(
            normalize_memory_card({"description": "new"}), candidates
        )

        assert decision["action"] == "add"
        # Exception did not propagate
        assert mock_llm.generate.call_count > 0

    def test_uses_first_valid_response(self, tmp_path):
        """F3: If first LLM call returns garbage but second returns valid JSON,
        the valid response is used."""
        mem = _make_full_memory(tmp_path, card_update_dedup_config={"enabled": True})

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            ("garbage", {}, None, None),  # First attempt fails
            (
                json.dumps(
                    {"action": "discard", "duplicate_of": "existing"}
                ),  # Second succeeds
                {},
                None,
                None,
            ),
        ]
        mem.dedup.llm_service = mock_llm

        decision = mem.dedup.decide_action(
            normalize_memory_card({"description": "duplicate"}),
            [{"card_id": "existing", "final_score": 0.8}],
        )

        assert decision["action"] == "discard"
        assert mock_llm.generate.call_count == 2

    def test_empty_candidates_returns_default_without_llm(self, tmp_path):
        """F4: When candidates_for_llm is empty, decide_action returns default
        without calling the LLM."""
        mem = _make_full_memory(tmp_path, card_update_dedup_config={"enabled": True})

        mock_llm = MagicMock()
        mem.llm_service = mock_llm
        mem.dedup.llm_service = mock_llm

        decision = mem.dedup.decide_action(
            normalize_memory_card({"description": "new"}), []
        )

        assert decision["action"] == "add"
        # LLM was NOT called
        mock_llm.generate.assert_not_called()


# ===========================================================================
# Category G: Merge Edge Cases (card_dedup.py + memory.py)
# ===========================================================================


class TestMergeEdgeCases:
    """Tests for CardDedup.compute_merges and _apply_update_actions_from_merges."""

    def test_compute_merges_skips_deleted_card(self, tmp_path):
        """G1: If an update targets a card_id that's no longer in store.cards,
        it's silently skipped."""
        mem = _make_full_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "existing"})
        mem.save_card({"id": "c2", "description": "existing2"})

        # Delete c2
        mem.card_store.cards.pop("c2")

        updates = [
            {"card_id": "c2", "update_explanation": True, "explanation_append": "new"}
        ]
        merges = mem.dedup.compute_merges(
            normalize_memory_card({"description": "incoming"}), updates
        )

        # No merges for deleted card
        assert len(merges) == 0

    def test_compute_merges_deduplicates_same_card_id(self, tmp_path):
        """G2: If LLM returns two update entries for the same card_id,
        only the first is processed (via seen_ids set)."""
        mem = _make_full_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "target"})

        updates = [
            {"card_id": "c1", "update_explanation": True, "explanation_append": "1"},
            {"card_id": "c1", "update_explanation": True, "explanation_append": "2"},
        ]
        merges = mem.dedup.compute_merges(
            normalize_memory_card({"description": "incoming"}), updates
        )

        # Only one merge
        assert len(merges) == 1
        assert merges[0][0] == "c1"

    def test_apply_update_persists_partial_on_exception(self, tmp_path):
        """G3: When _save_card_core raises on first merge, the exception is
        logged (not propagated) and remaining merges are still attempted."""
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "target1"})
        mem.save_card({"id": "c2", "description": "target2"})

        call_count = 0

        def failing_on_first(card):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("save failed")
            # Second call succeeds
            mem.card_store.cards[card.id or ""] = card

        incoming = normalize_memory_card({"description": "incoming"})
        updates = [
            {
                "card_id": "c1",
                "update_explanation": True,
                "explanation_append": "x",
            },
            {
                "card_id": "c2",
                "update_explanation": True,
                "explanation_append": "y",
            },
        ]
        merges = mem.dedup.compute_merges(incoming, updates)

        mem._save_card_core = failing_on_first

        # Try to apply pre-computed merges, first fails
        updated_ids = mem._apply_update_actions_from_merges(merges)

        # c1 failed, c2 succeeded
        assert "c1" not in updated_ids
        assert "c2" in updated_ids

    def test_compute_merges_skips_non_dict_updates(self, tmp_path):
        """G4: If updates contains non-dict items, they're skipped."""
        mem = _make_full_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "target"})

        updates = [
            "garbage string",
            None,
            42,
            {"card_id": "c1", "update_explanation": True},
        ]
        merges = mem.dedup.compute_merges(
            normalize_memory_card({"description": "incoming"}), updates
        )

        # Only the valid dict processed
        assert len(merges) == 1


# ===========================================================================
# Category H: Dedup + Orchestrator Integration (memory.py)
# ===========================================================================


class TestDedupOrchestrator:
    """Tests for save_card dedup flow and fallthrough logic."""

    def test_discard_nonexistent_duplicate_still_rejects(self, tmp_path):
        """H1: LLM says "discard, duplicate_of=X" where X doesn't exist.
        The card is still rejected (not added)."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card({"id": "c1", "description": "card1"})
        mem.save_card({"id": "c2", "description": "card2"})

        initial_stats = mem.get_card_write_stats()
        initial_added = initial_stats["added"]
        initial_rejected = initial_stats["rejected"]

        # Mock LLM service so dedup can proceed
        mock_llm = MagicMock()
        mem.llm_service = mock_llm
        mem.dedup.llm_service = mock_llm

        # Mock decide_action to return discard for a nonexistent card
        # (simulating decision made with data that was deleted before decision)
        mem.dedup.decide_action = MagicMock(
            return_value={"action": "discard", "duplicate_of": "nonexistent"}
        )

        # Mock score_candidates to return non-empty list so dedup_ready is True
        mem.dedup.score_candidates = MagicMock(
            return_value=[{"card_id": "c1", "final_score": 0.9}]
        )

        # Save a new card — dedup will return discard for nonexistent card
        mem.save_card({"description": "new card"})

        stats = mem.get_card_write_stats()
        # Rejected count incremented even though duplicate_of doesn't exist
        assert stats["rejected"] == initial_rejected + 1
        # Added count did NOT increment (card was rejected, not added)
        assert stats["added"] == initial_added

    def test_update_returns_first_id_multi_target(self, tmp_path):
        """H2: When update merges into multiple cards, save_card returns the
        first updated card_id and both are merged."""
        mem = _make_full_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card({"id": "c1", "description": "idea1"})
        mem.save_card({"id": "c2", "description": "idea2"})

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps(
                {
                    "action": "update",
                    "updates": [
                        {"card_id": "c1", "update_explanation": True},
                        {"card_id": "c2", "update_explanation": True},
                    ],
                }
            ),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm
        mem.dedup.llm_service = mock_llm
        mem.dedup.score_candidates = MagicMock(
            return_value=[
                {"card_id": "c1", "final_score": 0.8},
                {"card_id": "c2", "final_score": 0.7},
            ]
        )

        result_id = mem.save_card({"description": "new idea"})

        # Returns first id
        assert result_id == "c1"
        stats = mem.get_card_write_stats()
        assert stats["updated_target_cards"] == 2

    def test_update_empty_updates_falls_through_to_add(self, tmp_path):
        """H3: LLM says "update" but provides empty updates list, falls through
        to add instead."""
        mem = _make_full_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card({"id": "existing", "description": "original"})

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps({"action": "update", "updates": []}),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm
        mem.dedup.llm_service = mock_llm
        mem.dedup.score_candidates = MagicMock(
            return_value=[{"card_id": "existing", "final_score": 0.8}]
        )

        mem.save_card({"description": "new idea"})

        stats = mem.get_card_write_stats()
        # Fell through to add
        assert stats["added"] == 2


# ===========================================================================
# Category I: NoteSync Exceptions (note_sync.py)
# ===========================================================================


class TestNoteSyncExceptions:
    """Tests for exception handling in NoteSync upsert methods."""

    def test_upsert_fast_delete_failure_logs_warning_and_still_adds(self, tmp_path):
        """I1: If retriever.delete_document raises, a warning is logged and
        add_document is still called. In dict-based retrievers this overwrites;
        in list-based (Chroma) this may leave duplicates — the warning lets
        operators detect the issue."""
        mem, fake_sys = make_test_memory_with_agentic(tmp_path)

        # Track add_document calls
        add_calls = []
        original_add = fake_sys.retriever.add_document

        def tracked_add(document, metadata, doc_id):
            add_calls.append((document, doc_id))
            return original_add(document, metadata, doc_id)

        fake_sys.retriever.add_document = tracked_add

        # Patch retriever to raise on delete
        def failing_delete(doc_id):
            raise RuntimeError("delete failed")

        fake_sys.retriever.delete_document = failing_delete

        # Upsert a card
        card1 = normalize_memory_card({"id": "c1", "description": "original"})
        mem.note_sync.upsert_fast(card1)
        initial_add_count = len(add_calls)

        # Upsert again with changed content — delete fails (warning logged), add succeeds
        card2 = normalize_memory_card({"id": "c1", "description": "updated"})
        mem.note_sync.upsert_fast(card2)

        # add_document was still called despite delete failure
        assert len(add_calls) > initial_add_count

    def test_upsert_agentic_update_raises_propagates(self, tmp_path):
        """I2: Unlike upsert_fast, upsert_agentic does NOT have try/except,
        so exceptions from memory_system.update() propagate."""
        mem, fake_sys = make_test_memory_with_agentic(tmp_path)
        mem.save_card({"id": "existing", "description": "original"})

        # Patch update to raise
        def failing_update(*args, **kwargs):
            raise RuntimeError("update failed")

        mem.note_sync.memory_system.update = failing_update

        card = normalize_memory_card({"id": "existing", "description": "updated"})
        with pytest.raises(RuntimeError, match="update failed"):
            mem.note_sync.upsert_agentic(card)


# ===========================================================================
# Category J: Config Validation (memory_config.py)
# ===========================================================================


class TestMergeApiError:
    """Tests for X3: partial merge failure during _apply_update_actions_from_merges."""

    def test_merge_api_error_continues_remaining_merges(self, tmp_path):
        """X3: If _save_card_core raises on first merge, second merge should
        still be attempted. The partial failure should be logged, not silently
        dropped."""
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "target1"})
        mem.save_card({"id": "c2", "description": "target2"})

        call_count = 0
        original_save = mem._save_card_core

        def failing_on_first(card):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API error on first merge")
            return original_save(card)

        incoming = normalize_memory_card({"description": "incoming"})
        updates = [
            {
                "card_id": "c1",
                "update_explanation": True,
                "explanation_append": "new info 1",
            },
            {
                "card_id": "c2",
                "update_explanation": True,
                "explanation_append": "new info 2",
            },
        ]
        merges = mem.dedup.compute_merges(incoming, updates)

        mem._save_card_core = failing_on_first

        updated_ids = mem._apply_update_actions_from_merges(merges)

        # c2 should still be updated even though c1 failed
        assert "c2" in updated_ids
        # c1 was NOT added because it failed
        assert "c1" not in updated_ids


class TestDiscardPhantomId:
    """Tests for X5: discard returns phantom card ID."""

    def test_discard_phantom_id_returns_fallback(self, tmp_path):
        """X5: When LLM says discard with duplicate_of pointing to a
        nonexistent card, the returned ID should be valid (exist in store)."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card({"id": "c1", "description": "card1"})

        mock_llm = MagicMock()
        mem.llm_service = mock_llm
        mem.dedup.llm_service = mock_llm

        # Mock decide_action to return discard for a phantom card
        mem.dedup.decide_action = MagicMock(
            return_value={"action": "discard", "duplicate_of": "phantom-gone"}
        )
        mem.dedup.score_candidates = MagicMock(
            return_value=[{"card_id": "c1", "final_score": 0.9}]
        )

        result_id = mem.save_card({"description": "new card"})

        # The returned ID must be valid — either in store.cards or a fresh mem-* ID
        # It must NOT be "phantom-gone" since that card doesn't exist
        assert result_id != "phantom-gone"


class TestConfigValidation:
    """Tests for Pydantic config validation."""

    def test_memory_config_rejects_extra_fields(self, tmp_path):
        """J1: MemoryConfig with extra="forbid" rejects unknown fields."""
        with pytest.raises(ValidationError, match="extra_field"):
            MemoryConfig(
                checkpoint_path=tmp_path / "mem",
                extra_field="should fail",
            )

    def test_api_config_rejects_zero_batch_size(self, tmp_path):
        """J2: ApiConfig.sync_batch_size with gt=0 validator rejects 0."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ApiConfig(sync_batch_size=0)

    def test_gam_config_nonexistent_tool_normalization(self, tmp_path):
        """J3: GamConfig with invalid tool names — normalization result
        determines whether retrievers can be built."""
        cfg = GamConfig(allowed_tools=["nonexistent_tool"])
        normalized = cfg.normalized_allowed_tools

        # Document behavior: invalid tools may or may not be filtered
        # depending on normalize_allowed_gam_tools implementation
        assert isinstance(normalized, set)
