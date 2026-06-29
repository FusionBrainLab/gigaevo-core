"""Tests for Pydantic card models: MemoryCard, ProgramCard, AnyCard.

TDD RED phase: tests for structured card types replacing raw dicts.
"""

from gigaevo.memory.shared_memory.models import (
    AnyCard,
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

    def test_with_fitness(self):
        card = ProgramCard(id="p1", fitness=95.5, code="def f(): pass")
        assert card.fitness == 95.5
        assert card.code == "def f(): pass"

    def test_to_dict(self):
        card = ProgramCard(id="p1", program_id="prog-1", fitness=90.0)
        d = card.model_dump()
        assert d["id"] == "p1"
        assert d["category"] == "program"
        assert d["fitness"] == 90.0


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
