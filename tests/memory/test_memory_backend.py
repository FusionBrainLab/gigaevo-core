"""Tests for AmemGamMemory — the core memory API.

Tests use local-only mode (no API, no sync, no LLM, no agentic memory)
to pin down behavior for safe refactoring. Dedup tests mock the LLM.
"""

import json
from unittest.mock import MagicMock
import uuid

from gigaevo.memory.shared_memory.card_conversion import (
    MemoryCard,
    ProgramCard,
    is_program_card,
    normalize_allowed_gam_tools,
    normalize_gam_pipeline_mode,
    normalize_gam_top_k_by_tool,
)
from gigaevo.memory.shared_memory.utils import dedupe_keep_order, looks_like_uuid
from tests.fakes.agentic_memory import make_test_memory

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_memory(tmp_path, **overrides):
    return make_test_memory(tmp_path, **overrides)


def _make_card(**overrides):
    base = {
        "id": f"test-{uuid.uuid4().hex[:8]}",
        "description": "Test idea description",
        "task_description": "Solve the task",
        "task_description_summary": "Task summary",
    }
    base.update(overrides)
    return base


# ===========================================================================
# Init
# ===========================================================================


class TestAmemGamMemoryInit:
    def test_checkpoint_dir_created(self, tmp_path):
        mem = _make_memory(tmp_path)
        assert mem.config.checkpoint_path.exists()

    def test_api_disabled(self, tmp_path):
        mem = _make_memory(tmp_path)
        assert mem.api is None
        assert mem.api is None

    def test_stats_start_at_zero(self, tmp_path):
        mem = _make_memory(tmp_path)
        stats = mem.get_card_write_stats()
        assert all(v == 0 for v in stats.values())
        assert set(stats.keys()) == {
            "processed",
            "added",
            "rejected",
            "updated",
            "updated_target_cards",
        }

    def test_memory_cards_empty(self, tmp_path):
        mem = _make_memory(tmp_path)
        assert mem.card_store.cards == {}

    def test_memory_system_none_without_agentic_deps(self, tmp_path):
        """In CI, A_mem/GAM imports fail → memory_system is None."""
        mem = _make_memory(tmp_path)
        assert mem.memory_system is None

    def test_llm_service_none_without_api_key(self, tmp_path):
        mem = _make_memory(tmp_path)
        # Without OPENAI_API_KEY and no agentic deps, llm_service should be None
        assert mem.llm_service is None

    def test_research_agent_none(self, tmp_path):
        mem = _make_memory(tmp_path)
        assert mem.research_agent is None

    def test_index_file_path(self, tmp_path):
        mem = _make_memory(tmp_path)
        assert mem.config.index_file == mem.config.checkpoint_path / "api_index.json"


# ===========================================================================
# save_card / get_card
# ===========================================================================


class TestSaveCard:
    def test_returns_card_id(self, tmp_path):
        mem = _make_memory(tmp_path)
        card_id = mem.save_card(_make_card(id="c1"))
        assert card_id == "c1"

    def test_assigns_id_when_missing(self, tmp_path):
        mem = _make_memory(tmp_path)
        card_id = mem.save_card({"description": "no id"})
        assert card_id.startswith("mem-")

    def test_preserves_given_id(self, tmp_path):
        mem = _make_memory(tmp_path)
        card_id = mem.save_card(_make_card(id="my-custom-id"))
        assert card_id == "my-custom-id"

    def test_normalizes_card(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1"))
        stored = mem.get_card("c1")
        assert stored.description is not None
        assert stored.category is not None

    def test_increments_processed_and_added(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1"))
        stats = mem.get_card_write_stats()
        assert stats["processed"] == 1
        assert stats["added"] == 1

    def test_save_same_id_twice_increments_updated(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1", description="v1"))
        mem.save_card(_make_card(id="c1", description="v2"))
        stats = mem.get_card_write_stats()
        assert stats["processed"] == 2
        assert stats["updated"] == 1
        assert stats["added"] == 1
        # Second save should overwrite
        assert mem.get_card("c1").description == "v2"

    def test_multiple_cards(self, tmp_path):
        mem = _make_memory(tmp_path)
        for i in range(5):
            mem.save_card(_make_card(id=f"c{i}"))
        assert len(mem.card_store.cards) == 5
        stats = mem.get_card_write_stats()
        assert stats["processed"] == 5
        assert stats["added"] == 5

    def test_program_card_bypasses_dedup(self, tmp_path):
        """Program cards should always be added, never deduped."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        # Save a seed card first
        mem.save_card(_make_card(id="seed"))

        # Program card should bypass dedup even though dedup is enabled
        mem.save_card(
            {
                "category": "program",
                "program_id": "prog-1",
                "description": "Top program",
                "fitness": 95.0,
                "code": "def f(): pass",
            }
        )
        stats = mem.get_card_write_stats()
        assert stats["added"] == 2  # seed + program
        assert stats["rejected"] == 0

    def test_persists_to_index_file(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1", description="persisted"))
        assert mem.config.index_file.exists()
        data = json.loads(mem.config.index_file.read_text())
        assert "c1" in data["memory_cards"]


class TestGetCard:
    def test_existing(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1", description="hello"))
        card = mem.get_card("c1")
        assert card is not None
        assert card.description == "hello"

    def test_nonexistent(self, tmp_path):
        mem = _make_memory(tmp_path)
        assert mem.get_card("nonexistent") is None

    def test_returns_dict(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1"))
        card = mem.get_card("c1")
        assert card is not None

    def test_card_is_mutable_reference(self, tmp_path):
        """get_card returns a reference to the internal dict — mutations are visible."""
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1", description="original"))
        card = mem.get_card("c1")
        card.description = "mutated"
        # This documents current behavior: direct reference, not copy
        assert mem.get_card("c1").description == "mutated"


# ===========================================================================
# delete
# ===========================================================================


class TestDelete:
    def test_existing_returns_true(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1"))
        assert mem.delete("c1") is True

    def test_removes_card(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1"))
        mem.delete("c1")
        assert mem.get_card("c1") is None

    def test_nonexistent_returns_false(self, tmp_path):
        mem = _make_memory(tmp_path)
        assert mem.delete("nonexistent") is False

    def test_delete_persists(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1"))
        mem.delete("c1")
        data = json.loads(mem.config.index_file.read_text())
        assert "c1" not in data["memory_cards"]

    def test_delete_one_of_many(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1"))
        mem.save_card(_make_card(id="c2"))
        mem.save_card(_make_card(id="c3"))
        mem.delete("c2")
        assert mem.get_card("c1") is not None
        assert mem.get_card("c2") is None
        assert mem.get_card("c3") is not None


# ===========================================================================
# search (local)
# ===========================================================================


class TestSearchLocal:
    def test_empty_memory(self, tmp_path):
        mem = _make_memory(tmp_path)
        result = mem.search("anything")
        assert "No relevant memories found" in result

    def test_matches_by_description(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1", description="simulated annealing optimizer"))
        mem.save_card(_make_card(id="c2", description="genetic crossover mutation"))
        result = mem.search("annealing")
        assert "c1" in result
        # c2 should not match "annealing"
        assert "c2" not in result or result.index("c1") < result.index("c2")

    def test_matches_by_keywords(self, tmp_path):
        mem = _make_memory(tmp_path)
        card = _make_card(id="c1", description="something")
        # Save then inject keywords (normalize_memory_card produces keyword field)
        mem.save_card(card)
        # Directly modify to add keywords for search
        mem.card_store.cards["c1"] = mem.card_store.cards["c1"].model_copy(
            update={"keywords": ["optimization", "local-search"]}
        )
        result = mem.search("optimization")
        assert "c1" in result

    def test_no_match(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1", description="alpha beta gamma"))
        result = mem.search("zzzzunmatchable")
        assert "No relevant memories found" in result

    def test_respects_search_limit(self, tmp_path):
        mem = _make_memory(tmp_path, search_limit=2)
        for i in range(10):
            mem.save_card(
                _make_card(id=f"c{i}", description=f"idea about optimization {i}")
            )
        result = mem.search("optimization")
        # Count card IDs in result — should be at most 2
        found = [f"c{i}" for i in range(10) if f"c{i}" in result]
        assert len(found) <= 2

    def test_format_search_results_structure(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1", description="annealing idea"))
        result = mem.search("annealing")
        assert result.startswith("Query: annealing")
        assert "Top relevant memory cards:" in result

    def test_search_with_memory_state(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1", description="repair step for validation"))
        result = mem.search("repair", memory_state="validation context")
        assert "c1" in result


# ===========================================================================
# save() convenience method
# ===========================================================================


class TestSaveConvenience:
    def test_save_text(self, tmp_path):
        mem = _make_memory(tmp_path)
        card_id = mem.save("some text data")
        assert card_id.startswith("mem-")
        card = mem.get_card(card_id)
        assert card.description == "some text data"
        assert card.category == "general"


# ===========================================================================
# Dedup with mocked LLM
# ===========================================================================


class TestDedup:
    def test_dedup_disabled_always_adds(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="seed", description="seed idea"))
        mem.save_card(_make_card(description="new idea"))
        stats = mem.get_card_write_stats()
        assert stats["added"] == 2
        assert stats["rejected"] == 0

    def test_dedup_enabled_no_llm_falls_back_to_add(self, tmp_path):
        """When dedup is enabled but LLM is unavailable, cards are still added."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        assert mem.llm_service is None
        mem.save_card(_make_card(id="seed"))
        mem.save_card(_make_card(description="new"))
        stats = mem.get_card_write_stats()
        assert stats["added"] == 2
        assert stats["rejected"] == 0

    def test_dedup_discard_action(self, tmp_path):
        """Mock LLM returns discard → card is rejected."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card(_make_card(id="existing", description="original idea"))

        # Mock LLM service
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps({"action": "discard", "duplicate_of": "existing"}),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm

        # Mock _score_retrieved_candidates to return a synthetic candidate
        mem.dedup.score_candidates = MagicMock(
            return_value=[{"card_id": "existing", "score": 0.9}]
        )

        card_id = mem.save_card(_make_card(description="duplicate idea"))
        assert card_id == "existing"
        stats = mem.get_card_write_stats()
        assert stats["rejected"] == 1

    def test_dedup_add_action(self, tmp_path):
        """Mock LLM returns add → card is added normally."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card(_make_card(id="existing"))

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps({"action": "add"}),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm
        mem.dedup.score_candidates = MagicMock(
            return_value=[{"card_id": "existing", "score": 0.3}]
        )

        mem.save_card(_make_card(description="new unique idea"))
        stats = mem.get_card_write_stats()
        assert stats["added"] == 2  # existing + new

    def test_dedup_only_triggers_with_existing_cards(self, tmp_path):
        """Dedup requires card_store.cards to be non-empty."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mock_llm = MagicMock()
        mem.llm_service = mock_llm

        # First card — memory_cards is empty, should NOT call LLM
        mem.save_card(_make_card(id="first"))
        mock_llm.generate.assert_not_called()


# ===========================================================================
# Index persistence / reload
# ===========================================================================


class TestIndexPersistence:
    def test_persist_and_reload(self, tmp_path):
        mem1 = _make_memory(tmp_path)
        mem1.save_card(_make_card(id="c1", description="idea one"))
        mem1.save_card(_make_card(id="c2", description="idea two"))

        # Create new instance from same checkpoint
        mem2 = _make_memory(tmp_path)
        assert mem2.get_card("c1") is not None
        assert mem2.get_card("c1").description == "idea one"
        assert mem2.get_card("c2") is not None

    def test_malformed_json_handled(self, tmp_path):
        mem_path = tmp_path / "mem"
        mem_path.mkdir(parents=True)
        index_file = mem_path / "api_index.json"
        index_file.write_text("not valid json {{{")

        # Should not crash
        mem = _make_memory(tmp_path)
        assert mem.card_store.cards == {}

    def test_empty_index_file(self, tmp_path):
        mem_path = tmp_path / "mem"
        mem_path.mkdir(parents=True)
        index_file = mem_path / "api_index.json"
        index_file.write_text("{}")

        mem = _make_memory(tmp_path)
        assert mem.card_store.cards == {}

    def test_index_with_entity_mappings(self, tmp_path):
        """Verify entity_by_card_id and card_id_by_entity round-trip."""
        mem1 = _make_memory(tmp_path)
        mem1.save_card(_make_card(id="c1"))
        # Manually add entity mapping (normally done by API mode)
        mem1.card_store.entity_by_card_id["c1"] = "entity-uuid"
        mem1.card_store.card_id_by_entity["entity-uuid"] = "c1"
        mem1.card_store.persist()

        mem2 = _make_memory(tmp_path)
        assert mem2.card_store.entity_by_card_id.get("c1") == "entity-uuid"
        assert mem2.card_store.card_id_by_entity.get("entity-uuid") == "c1"


# ===========================================================================
# Static helpers
# ===========================================================================


class TestStaticHelpers:
    def test_looks_like_uuid_valid(self):
        assert looks_like_uuid("12345678-1234-5678-1234-567812345678")

    def test_looks_like_uuid_hex(self):
        assert looks_like_uuid("12345678123456781234567812345678")

    def test_looks_like_uuid_invalid(self):
        assert not looks_like_uuid("not-a-uuid")

    def test_looks_like_uuid_empty(self):
        assert not looks_like_uuid("")

    def test_is_program_card_by_category(self):
        assert is_program_card(ProgramCard(id="p1"))

    def test_is_program_card_by_program_id(self):
        assert is_program_card(ProgramCard(id="p1", program_id="p1"))

    def test_is_program_card_false_for_general(self):
        assert not is_program_card(MemoryCard(id="c1"))

    def test_is_program_card_false_empty(self):
        assert not is_program_card(MemoryCard(id="c1"))

    def test_dedupe_keep_order(self):
        assert dedupe_keep_order(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_dedupe_keep_order_strips_empty(self):
        assert dedupe_keep_order(["a", "", "  ", "b"]) == ["a", "b"]

    def test_dedupe_keep_order_strips_whitespace(self):
        assert dedupe_keep_order([" a ", "a"]) == ["a"]

    def test_normalize_allowed_gam_tools_defaults(self):
        result = normalize_allowed_gam_tools(None)
        assert "keyword" in result
        assert "vector" in result

    def test_normalize_allowed_gam_tools_custom(self):
        result = normalize_allowed_gam_tools(["keyword", "page_index"])
        assert result == {"keyword", "page_index"}

    def test_normalize_allowed_gam_tools_vector_expands(self):
        result = normalize_allowed_gam_tools(["vector"])
        # "vector" should expand to all vector-backed tools
        assert "vector_description" in result
        assert "vector_task_description" in result

    def test_normalize_allowed_gam_tools_invalid_ignored(self):
        result = normalize_allowed_gam_tools(["bogus_tool"])
        # Falls back to defaults when all custom tools are invalid
        assert "keyword" in result

    def test_normalize_gam_pipeline_mode_valid(self):
        assert normalize_gam_pipeline_mode("default") == "default"
        assert normalize_gam_pipeline_mode("experimental") == "experimental"

    def test_normalize_gam_pipeline_mode_invalid(self):
        assert normalize_gam_pipeline_mode("bogus") == "default"

    def test_normalize_gam_pipeline_mode_none(self):
        assert normalize_gam_pipeline_mode(None) == "default"

    def test_normalize_gam_top_k_default(self):
        result = normalize_gam_top_k_by_tool(None)
        assert result["keyword"] == 5
        assert result["vector"] == 5

    def test_normalize_gam_top_k_custom(self):
        result = normalize_gam_top_k_by_tool({"keyword": 10})
        assert result["keyword"] == 10
        assert result["vector"] == 5  # unchanged

    def test_normalize_gam_top_k_invalid_value_ignored(self):
        result = normalize_gam_top_k_by_tool({"keyword": "abc"})
        assert result["keyword"] == 5  # default preserved

    def test_normalize_gam_top_k_zero_ignored(self):
        result = normalize_gam_top_k_by_tool({"keyword": 0})
        assert result["keyword"] == 5  # zero is not > 0


# ===========================================================================
# rebuild
# ===========================================================================


class TestRebuild:
    def test_rebuild_no_crash_without_agentic(self, tmp_path):
        """rebuild() should be safe when memory_system is None."""
        mem = _make_memory(tmp_path)
        mem.save_card(_make_card(id="c1"))
        mem.rebuild()  # Should not crash
        assert mem.get_card("c1") is not None


# ===========================================================================
# get_card_write_stats
# ===========================================================================


class TestGetCardWriteStats:
    def test_returns_copy(self, tmp_path):
        mem = _make_memory(tmp_path)
        stats1 = mem.get_card_write_stats()
        stats1["processed"] = 999
        stats2 = mem.get_card_write_stats()
        assert stats2["processed"] == 0  # not mutated
