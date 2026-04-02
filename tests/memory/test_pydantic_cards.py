"""Tests for Pydantic card models: MemoryCard, ProgramCard, AnyCard.

TDD RED phase: tests for structured card types replacing raw dicts.
"""

from gigaevo.memory.shared_memory.models import (
    AnyCard,
    ConnectedIdea,
    MemoryCard,
    ProgramCard,
)


class TestProgramCard:
    def test_minimal(self):
        card = ProgramCard(id="p1")
        assert card.id == "p1"
        assert card.category == "program"
        assert card.fitness is None
        assert card.code == ""
        assert card.connected_ideas == []

    def test_with_fitness(self):
        card = ProgramCard(id="p1", fitness=95.5, code="def f(): pass")
        assert card.fitness == 95.5
        assert card.code == "def f(): pass"

    def test_connected_ideas_as_dicts(self):
        card = ProgramCard(
            id="p1",
            connected_ideas=[{"idea_id": "i1", "description": "SA"}],
        )
        assert len(card.connected_ideas) == 1

    def test_connected_ideas_as_models(self):
        card = ProgramCard(
            id="p1",
            connected_ideas=[ConnectedIdea(idea_id="i1", description="SA")],
        )
        assert card.connected_ideas[0].idea_id == "i1"

    def test_to_dict(self):
        card = ProgramCard(id="p1", program_id="prog-1", fitness=90.0)
        d = card.model_dump()
        assert d["id"] == "p1"
        assert d["category"] == "program"
        assert d["fitness"] == 90.0


class TestMemoryCardStrategyFlexible:
    """Strategy was previously Literal — now str for flexibility."""

    def test_any_string_strategy(self):
        card = MemoryCard(id="c1", strategy="exploration")
        assert card.strategy == "exploration"

    def test_empty_strategy(self):
        card = MemoryCard(id="c1", strategy="")
        assert card.strategy == ""

    def test_custom_strategy(self):
        card = MemoryCard(id="c1", strategy="custom_approach")
        assert card.strategy == "custom_approach"


class TestAnyCardUnion:
    def test_general_card(self):
        card: AnyCard = MemoryCard(id="c1", description="idea")
        assert isinstance(card, MemoryCard)

    def test_program_card(self):
        card: AnyCard = ProgramCard(id="p1", program_id="prog-1")
        assert isinstance(card, ProgramCard)

    def test_both_have_common_fields(self):
        general: AnyCard = MemoryCard(id="c1", description="d", task_description="t")
        program: AnyCard = ProgramCard(id="p1", description="d", task_description="t")
        assert general.id == "c1"
        assert program.id == "p1"
        assert general.description == "d"
        assert program.description == "d"


class TestConnectedIdea:
    def test_basic(self):
        ci = ConnectedIdea(idea_id="i1", description="SA optimization")
        assert ci.idea_id == "i1"

    def test_extra_fields_allowed(self):
        ci = ConnectedIdea(idea_id="i1", description="d", score=0.9)
        assert ci.idea_id == "i1"
