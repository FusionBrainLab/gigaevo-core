"""Tests for Pydantic models in gigaevo.memory.shared_memory.models.

Pin down validation behavior: required fields, defaults, extra="forbid".
"""

import pytest
from pydantic import ValidationError

from gigaevo.memory.shared_memory.models import (
    LocalMemorySnapshot,
    MemoryCard,
    MemoryCardExplanation,
)


# ===========================================================================
# MemoryCardExplanation
# ===========================================================================


class TestMemoryCardExplanation:
    def test_defaults(self):
        e = MemoryCardExplanation()
        assert e.explanations == []
        assert e.summary == ""

    def test_with_values(self):
        e = MemoryCardExplanation(explanations=["a", "b"], summary="sum")
        assert e.explanations == ["a", "b"]
        assert e.summary == "sum"

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            MemoryCardExplanation(foo="bar")


# ===========================================================================
# MemoryCard
# ===========================================================================


class TestMemoryCard:
    def test_minimal_valid(self):
        c = MemoryCard(id="x", description="d")
        assert c.id == "x"
        assert c.description == "d"

    def test_defaults(self):
        c = MemoryCard(id="x", description="d")
        assert c.category == "general"
        assert c.task_description == ""
        assert c.task_description_summary == ""
        assert c.strategy is None
        assert c.last_generation == 0
        assert c.programs == []
        assert c.aliases == []
        assert c.keywords == []
        assert c.evolution_statistics == {}
        assert c.explanation.explanations == []
        assert c.explanation.summary == ""
        assert c.works_with == []
        assert c.links == []
        assert c.usage == {}

    def test_full_card(self):
        c = MemoryCard(
            id="test",
            description="desc",
            category="insight",
            task_description="td",
            task_description_summary="tds",
            strategy="exploration",
            last_generation=5,
            programs=["p1"],
            aliases=["a1"],
            keywords=["k1"],
            evolution_statistics={"x": 1},
            explanation=MemoryCardExplanation(explanations=["e"], summary="s"),
            works_with=["w1"],
            links=["l1"],
            usage={"u": 1},
        )
        assert c.strategy == "exploration"
        assert c.last_generation == 5

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            MemoryCard(description="d")

    def test_missing_description_raises(self):
        with pytest.raises(ValidationError):
            MemoryCard(id="x")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            MemoryCard(id="x", description="d", unknown_field="val")

    def test_strategy_valid_values(self):
        for s in ("exploration", "exploitation", "hybrid"):
            c = MemoryCard(id="x", description="d", strategy=s)
            assert c.strategy == s

    def test_strategy_none_valid(self):
        c = MemoryCard(id="x", description="d", strategy=None)
        assert c.strategy is None

    def test_strategy_invalid_raises(self):
        with pytest.raises(ValidationError):
            MemoryCard(id="x", description="d", strategy="random")

    def test_list_fields_are_independent_instances(self):
        """Default factory creates new lists per instance."""
        c1 = MemoryCard(id="a", description="d")
        c2 = MemoryCard(id="b", description="d")
        c1.programs.append("p1")
        assert c2.programs == []

    def test_dict_fields_are_independent_instances(self):
        c1 = MemoryCard(id="a", description="d")
        c2 = MemoryCard(id="b", description="d")
        c1.evolution_statistics["x"] = 1
        assert c2.evolution_statistics == {}


# ===========================================================================
# LocalMemorySnapshot
# ===========================================================================


class TestLocalMemorySnapshot:
    def test_empty(self):
        s = LocalMemorySnapshot()
        assert s.memory_cards == {}

    def test_with_cards(self):
        card = MemoryCard(id="c1", description="desc")
        s = LocalMemorySnapshot(memory_cards={"c1": card})
        assert "c1" in s.memory_cards
        assert s.memory_cards["c1"].description == "desc"

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            LocalMemorySnapshot(extra="bad")
