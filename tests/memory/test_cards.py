"""Card domain-model invariants: kind-gating and the absorbed-id self-loop.

The one Card type drives insight-vs-exemplar behavior off ``kind`` alone, so the
kind gate (exemplar fields only on ``kind=program``) and the absorbed-id
self-loop guard are the model's load-bearing invariants.
"""

from __future__ import annotations

import pytest

from gigaevo.memory.cards import Card, CardKind


def test_insight_card_rejects_exemplar_fields() -> None:
    for field in ({"code": "x = 1"}, {"program_id": "p1"}, {"fitness": 0.5}):
        with pytest.raises(ValueError):
            Card(id="mem-1", kind=CardKind.INSIGHT, **field)


def test_program_card_requires_program_id() -> None:
    with pytest.raises(ValueError):
        Card(id="program-1", kind=CardKind.PROGRAM)


def test_program_card_accepts_exemplar_fields() -> None:
    card = Card(
        id="program-1",
        kind=CardKind.PROGRAM,
        program_id="p1",
        code="x = 1",
        fitness=0.5,
    )
    assert card.program_id == "p1"
    assert card.code == "x = 1"
    assert card.fitness == 0.5


def test_card_cannot_absorb_its_own_id() -> None:
    with pytest.raises(ValueError):
        Card(id="mem-1", absorbed_ids=("mem-2", "mem-1"))


def test_card_absorbs_other_ids() -> None:
    card = Card(id="mem-1", absorbed_ids=("mem-2", "mem-3"))
    assert card.absorbed_ids == ("mem-2", "mem-3")
