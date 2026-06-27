"""Contract tests for the memory system.

These tests pin the exact shape, types, and invariants that downstream code
depends on. They are designed to BREAK loudly if a refactor changes any
behavioral contract. Each test documents WHY the contract matters.
"""

import json

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.memory.core import MemorySelection
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.models import (
    MemoryCard,
    ProgramCard,
)
from tests.fakes.agentic_memory import make_test_memory


def _make_memory(tmp_path, **overrides):
    return make_test_memory(tmp_path, **overrides)


# ===========================================================================
# Contract 1: normalize_memory_card output types
# ===========================================================================

_GENERAL_CARD_FIELDS = frozenset(MemoryCard.model_fields.keys())
_PROGRAM_CARD_FIELDS = frozenset(ProgramCard.model_fields.keys())


class TestNormalizeCardContract:
    """Pin the output type of normalize_memory_card."""

    def test_general_card_returns_memory_card(self):
        card = normalize_memory_card({"description": "test"})
        assert isinstance(card, MemoryCard)
        assert set(MemoryCard.model_fields.keys()) == _GENERAL_CARD_FIELDS

    def test_program_card_returns_program_card(self):
        card = normalize_memory_card({"category": "program", "program_id": "p1"})
        assert isinstance(card, ProgramCard)
        assert set(ProgramCard.model_fields.keys()) == _PROGRAM_CARD_FIELDS

    def test_general_card_field_types(self):
        card = normalize_memory_card({"id": "c1", "description": "d"})
        assert isinstance(card, MemoryCard)
        assert isinstance(card.id, str)
        assert isinstance(card.category, str)
        assert isinstance(card.description, str)
        assert isinstance(card.task_description, str)
        assert isinstance(card.task_description_summary, str)
        assert isinstance(card.programs, list)
        assert isinstance(card.keywords, list)
        assert card.gain_events is None

    def test_program_card_field_types(self):
        card = normalize_memory_card(
            {"category": "program", "program_id": "p1", "fitness": 90.0}
        )
        assert isinstance(card, ProgramCard)
        assert card.category == "program"
        assert isinstance(card.program_id, str)
        assert isinstance(card.description, str)
        assert isinstance(card.code, str)
        assert card.fitness is None or isinstance(card.fitness, float)


# ===========================================================================
# Contract 2: save → get roundtrip preserves data
# ===========================================================================


class TestSaveGetRoundtrip:
    """save_card → get_card must preserve all meaningful fields."""

    def test_general_card_roundtrip(self, tmp_path):
        mem = _make_memory(tmp_path)
        original = {
            "id": "c1",
            "description": "Use simulated annealing for local search",
            "task_description": "Solve TSP efficiently",
            "task_description_summary": "TSP solver",
            "programs": ["prog-1", "prog-2"],
            "keywords": ["SA", "local-search"],
        }
        mem.save_card(original)
        stored = mem.get_card("c1")
        assert isinstance(stored, MemoryCard)
        assert stored.id == "c1"
        assert stored.description == original["description"]
        assert stored.task_description == original["task_description"]
        assert stored.task_description_summary == original["task_description_summary"]
        assert stored.programs == original["programs"]
        assert stored.keywords == original["keywords"]

    def test_program_card_roundtrip(self, tmp_path):
        mem = _make_memory(tmp_path)
        original = {
            "id": "prog-1",
            "category": "program",
            "program_id": "prog-1",
            "description": "Top evolved program",
            "fitness": 95.5,
            "code": "def solve(x):\n    return sorted(x)\n",
            "task_description": "Solve TSP",
            "task_description_summary": "TSP",
        }
        mem.save_card(original)
        stored = mem.get_card("prog-1")
        assert isinstance(stored, ProgramCard)
        assert stored.category == "program"
        assert stored.program_id == "prog-1"
        assert stored.fitness == 95.5
        assert stored.code == original["code"]

    def test_persist_reload_roundtrip(self, tmp_path):
        mem1 = _make_memory(tmp_path)
        mem1.save_card(
            {
                "id": "c1",
                "description": "test idea",
                "keywords": ["k1", "k2"],
            }
        )
        mem2 = _make_memory(tmp_path)
        stored = mem2.get_card("c1")
        assert stored.description == "test idea"
        assert stored.keywords == ["k1", "k2"]


# ===========================================================================
# Contract 3: search() output format
# ===========================================================================


class TestSearchOutputContract:
    def test_no_results_format(self, tmp_path):
        mem = _make_memory(tmp_path)
        result = mem.search("anything")
        assert "No relevant memories found" in result

    def test_results_format_has_query_line(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(
            {"id": "c1", "description": "annealing idea", "keywords": ["annealing"]}
        )
        result = mem.search("annealing")
        assert result.startswith("Query: annealing")

    def test_results_contain_card_ids(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "idea-abc-123", "description": "unique approach"})
        result = mem.search("unique approach")
        assert "idea-abc-123" in result


# ===========================================================================
# Contract 4: api_index.json persistence format
# ===========================================================================


class TestIndexPersistenceContract:
    def test_index_has_required_top_level_keys(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "test"})
        data = json.loads(mem.config.index_file.read_text())
        assert "memory_cards" in data
        assert "entity_by_card_id" in data
        assert "entity_version_by_entity" in data

    def test_memory_cards_indexed_by_id(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "test"})
        data = json.loads(mem.config.index_file.read_text())
        assert "c1" in data["memory_cards"]
        assert data["memory_cards"]["c1"]["description"] == "test"

    def test_index_card_is_serialized_dict(self, tmp_path):
        """Persisted cards are JSON dicts (model_dump output)."""
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "test"})
        data = json.loads(mem.config.index_file.read_text())
        card_data = data["memory_cards"]["c1"]
        assert isinstance(card_data, dict)
        assert "id" in card_data
        assert "description" in card_data

    def test_index_backward_compatible_load(self, tmp_path):
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir(parents=True)
        index = {
            "memory_cards": {
                "c1": {
                    "id": "c1",
                    "description": "old format card",
                    "category": "general",
                },
            },
            "entity_by_card_id": {},
            "entity_version_by_entity": {},
        }
        (mem_dir / "api_index.json").write_text(json.dumps(index))
        mem = _make_memory(tmp_path)
        card = mem.get_card("c1")
        assert card is not None
        assert card.description == "old format card"
        assert card.programs == []


# ===========================================================================
# Contract 5: MemorySelection shape
# ===========================================================================


class TestMemorySelectionContract:
    def test_memory_selection_shape(self):
        sel = MemorySelection(cards=["1. idea"], card_ids=["id-1"])
        assert isinstance(sel.cards, list)
        assert isinstance(sel.card_ids, list)


# ===========================================================================
# Contract 8: mutation metadata keys
# ===========================================================================


class TestMutationMetadataKeysContract:
    def test_metadata_key_values(self):
        assert MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY == "memory_selected_idea_ids"
