"""Tests for memory_platform serialization: Pydantic → dict → JSON.

Verifies that memory_platform properly handles Pydantic model inputs
from write_pipeline.py through the full save_card → _persist_index flow.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from gigaevo.memory.shared_memory.models import (
    ConnectedIdea,
    MemoryCard,
    MemoryCardExplanation,
    ProgramCard,
)
from gigaevo.memory_platform.shared_memory.memory import (
    AmemGamMemory,
    normalize_memory_card,
)


# ---------------------------------------------------------------------------
# normalize_memory_card: Pydantic → dict conversion
# ---------------------------------------------------------------------------


class TestNormalizeMemoryCardPydanticInput:
    """Pydantic models from write_pipeline must be flattened to plain dicts."""

    def test_program_card_with_connected_ideas(self):
        card = ProgramCard(
            id="prog-1",
            program_id="p1",
            description="Top evolved program",
            fitness=95.0,
            connected_ideas=[
                ConnectedIdea(card_id="idea-1", description="Use annealing"),
                ConnectedIdea(card_id="idea-2", description="Chunking"),
            ],
        )
        result = normalize_memory_card(card)

        assert isinstance(result, dict)
        assert result["id"] == "prog-1"
        assert result["category"] == "program"
        for ci in result["connected_ideas"]:
            assert isinstance(ci, dict), f"Expected dict, got {type(ci)}"
        json.dumps(result)

    def test_memory_card_with_explanation(self):
        card = MemoryCard(
            id="idea-1",
            description="Use simulated annealing",
            explanation=MemoryCardExplanation(
                explanations=["Found this pattern"],
                summary="SA works well",
            ),
        )
        result = normalize_memory_card(card)

        assert isinstance(result, dict)
        assert isinstance(result["explanation"], dict)
        assert result["explanation"]["summary"] == "SA works well"
        assert result["explanation"]["explanations"] == ["Found this pattern"]
        json.dumps(result)

    def test_plain_dict_still_works(self):
        card = {"id": "c1", "description": "plain dict card", "category": "general"}
        result = normalize_memory_card(card)

        assert isinstance(result, dict)
        assert result["id"] == "c1"
        json.dumps(result)

    def test_program_card_roundtrip(self):
        card = ProgramCard(
            id="prog-2",
            program_id="p2",
            description="Evolved solver",
            fitness=88.5,
            code="def solve(): pass",
            connected_ideas=[
                ConnectedIdea(card_id="i1", description="idea one"),
            ],
            keywords=["solver", "evolution"],
        )
        result = normalize_memory_card(card)
        text = json.dumps(result, ensure_ascii=True, indent=2)
        parsed = json.loads(text)
        assert parsed["id"] == "prog-2"
        assert parsed["connected_ideas"][0]["card_id"] == "i1"

    def test_memory_card_roundtrip(self):
        card = MemoryCard(
            id="idea-2",
            description="Use gradient-free optimization",
            task_description_summary="TSP solver",
            keywords=["optimization", "TSP"],
            explanation=MemoryCardExplanation(
                explanations=["Tried CMA-ES", "Tried DE"],
                summary="Gradient-free methods outperform",
            ),
            works_with=["idea-3"],
            links=["https://example.com"],
        )
        result = normalize_memory_card(card)
        text = json.dumps(result, ensure_ascii=True, indent=2)
        parsed = json.loads(text)
        assert parsed["explanation"]["summary"] == "Gradient-free methods outperform"
        assert len(parsed["explanation"]["explanations"]) == 2

    def test_none_input(self):
        result = normalize_memory_card(None)
        assert isinstance(result, dict)
        json.dumps(result)


# ---------------------------------------------------------------------------
# Full flow: save_card → _persist_index (JSON serialization)
# ---------------------------------------------------------------------------


def _make_platform_memory(tmp_path):
    """Create AmemGamMemory with mocked network dependencies."""
    with patch.object(AmemGamMemory, "__init__", lambda self, **kw: None):
        mem = AmemGamMemory.__new__(AmemGamMemory)

    from pathlib import Path

    mem.checkpoint_dir = Path(tmp_path)
    mem.index_file = mem.checkpoint_dir / "platform_index.json"
    mem.gam_store_dir = mem.checkpoint_dir / "gam_shared" / "platform_store"
    mem.base_url = "http://test:8000"
    mem.use_api = True
    mem.namespace = "default"
    mem.author = None
    mem.channel = "latest"
    mem.search_limit = 5
    mem.enable_llm_synthesis = False
    mem.enable_memory_evolution = False
    mem.enable_llm_card_enrichment = False
    mem.rebuild_interval = 999
    mem.enable_bm25 = False
    mem.sync_batch_size = 100
    mem.allowed_gam_tools = set()
    mem.gam_top_k_by_tool = {}
    mem.gam_pipeline_mode = "default"
    mem.remote_vector_search_type = "vector"
    mem.remote_hybrid_weights = (0.4, 0.6)

    from gigaevo.memory.shared_memory.card_update_dedup import CardUpdateDedupConfig

    mem.card_update_dedup_config = CardUpdateDedupConfig()

    mem.memory_cards = {}
    mem.entity_by_card_id = {}
    mem.card_id_by_entity = {}
    mem.entity_version_by_entity = {}
    mem.memory_ids = set()
    mem.card_write_stats = {
        "processed": 0,
        "added": 0,
        "rejected": 0,
        "updated": 0,
        "updated_target_cards": 0,
    }
    mem._dedup_retrievers = None
    mem.research_agent = None
    mem._warned_missing_card_update_llm = False
    mem._iters_after_rebuild = 0
    mem.llm_service = None
    mem.memory_system = None
    mem.generator = None

    # Mock the API client so save_card doesn't make network calls
    mock_client = MagicMock()
    mock_ref = MagicMock()
    mock_ref.entity_id = "eid-1"
    mock_ref.version_id = "v1"
    mock_client.save_memory_card.return_value = mock_ref
    mem.client = mock_client

    return mem


class TestSaveCardPydanticFlow:
    """Test full save_card → _persist_index flow with Pydantic inputs."""

    def test_save_program_card_persists_as_json(self, tmp_path):
        mem = _make_platform_memory(tmp_path)
        card = ProgramCard(
            id="prog-1",
            program_id="p1",
            description="Top evolved program",
            fitness=95.0,
            connected_ideas=[
                ConnectedIdea(card_id="idea-1", description="Use annealing"),
            ],
        )
        card_id = mem.save_card(card)

        # Index file must be valid JSON (no Pydantic objects)
        assert mem.index_file.exists()
        payload = json.loads(mem.index_file.read_text())
        assert card_id in payload["memory_cards"]
        stored = payload["memory_cards"][card_id]
        assert isinstance(stored["connected_ideas"][0], dict)
        assert stored["connected_ideas"][0]["card_id"] == "idea-1"

    def test_save_memory_card_persists_explanation(self, tmp_path):
        mem = _make_platform_memory(tmp_path)
        card = MemoryCard(
            id="idea-1",
            description="Use simulated annealing",
            explanation=MemoryCardExplanation(
                explanations=["Pattern found in runs"],
                summary="SA works well",
            ),
        )
        card_id = mem.save_card(card)

        payload = json.loads(mem.index_file.read_text())
        stored = payload["memory_cards"][card_id]
        assert isinstance(stored["explanation"], dict)
        assert stored["explanation"]["summary"] == "SA works well"

    def test_save_multiple_cards_all_serializable(self, tmp_path):
        mem = _make_platform_memory(tmp_path)
        cards = [
            ProgramCard(
                id="prog-1",
                program_id="p1",
                description="program",
                fitness=90.0,
                connected_ideas=[
                    ConnectedIdea(card_id="i1", description="d1"),
                ],
            ),
            MemoryCard(
                id="idea-1",
                description="idea card",
                explanation=MemoryCardExplanation(
                    explanations=["e1", "e2"],
                    summary="summary",
                ),
                keywords=["kw1"],
            ),
        ]
        for card in cards:
            mem.save_card(card)

        payload = json.loads(mem.index_file.read_text())
        assert len(payload["memory_cards"]) == 2
        # Re-serialize to verify no hidden Pydantic objects
        json.dumps(payload)

    def test_card_to_backend_content_sends_clean_dict(self, tmp_path):
        mem = _make_platform_memory(tmp_path)
        card = ProgramCard(
            id="prog-1",
            program_id="p1",
            description="test",
            fitness=50.0,
            connected_ideas=[
                ConnectedIdea(card_id="c1", description="d1"),
            ],
        )
        mem.save_card(card)

        # Verify what was sent to the API
        call_args = mem.client.save_memory_card.call_args
        content = call_args[0][0]  # first positional arg
        # Content must be JSON-serializable (API sends it as HTTP body)
        json.dumps(content)
        assert isinstance(content["connected_ideas"][0], dict)

    def test_persist_index_reload_roundtrip(self, tmp_path):
        """Save cards, reload index, verify data integrity."""
        mem = _make_platform_memory(tmp_path)
        card = ProgramCard(
            id="prog-1",
            program_id="p1",
            description="test program",
            fitness=80.0,
            connected_ideas=[
                ConnectedIdea(card_id="i1", description="linked idea"),
            ],
        )
        mem.save_card(card)

        # Create new instance, load from persisted index
        mem2 = _make_platform_memory(tmp_path)
        mem2._load_index()

        assert "prog-1" in mem2.memory_cards
        stored = mem2.memory_cards["prog-1"]
        assert isinstance(stored, dict)
        assert stored["connected_ideas"][0]["card_id"] == "i1"
        # Must still be serializable after reload
        json.dumps(stored)
