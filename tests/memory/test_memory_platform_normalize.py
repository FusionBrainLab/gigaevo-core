"""Tests for memory_platform.normalize_memory_card: Pydantic → dict serialization.

Verifies that memory_platform's normalize_memory_card properly handles
Pydantic model inputs (the exact data flow from write_pipeline.py).
"""

import json

from gigaevo.memory.shared_memory.models import (
    ConnectedIdea,
    MemoryCard,
    MemoryCardExplanation,
    ProgramCard,
)
from gigaevo.memory_platform.shared_memory.memory import normalize_memory_card


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
