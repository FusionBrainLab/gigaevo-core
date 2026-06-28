"""Tests for Pydantic models in gigaevo.memory.shared_memory.models.

Pin down validation behavior: required fields, defaults, extra="forbid".
"""

from pydantic import ValidationError
import pytest

from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.shared_memory.models import (
    MemoryCard,
)

# ===========================================================================
# MemoryCard
# ===========================================================================


class TestMemoryCard:
    def test_minimal_valid(self):
        c = MemoryCard(id="x", description="d")
        assert c.id == "x"
        assert c.description == "d"

    def test_defaults(self):
        c = MemoryCard(id="x")
        assert c.category == "general"
        assert c.description == ""
        assert c.task_description == ""
        assert c.task_description_summary == ""
        assert c.programs == []
        assert c.keywords == []
        assert c.gain_events is None

    def test_full_card(self):
        c = MemoryCard(
            id="test",
            description="desc",
            category="insight",
            task_description="td",
            task_description_summary="tds",
            programs=["p1"],
            keywords=["k1"],
            gain_events=[
                ContextualGain(
                    context=DecisionContext(parent_metrics={"min_area": 0.5}),
                    gain=0.01,
                )
            ],
        )
        assert c.category == "insight"
        assert c.programs == ["p1"]
        assert len(c.gain_events) == 1

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            MemoryCard(description="d")

    def test_description_defaults_to_empty(self):
        c = MemoryCard(id="x")
        assert c.description == ""

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            MemoryCard(id="x", description="d", unknown_field="val")

    def test_list_fields_are_independent_instances(self):
        """Default factory creates new lists per instance."""
        c1 = MemoryCard(id="a", description="d")
        c2 = MemoryCard(id="b", description="d")
        c1.programs.append("p1")
        assert c2.programs == []
