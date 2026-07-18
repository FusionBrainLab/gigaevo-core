"""Card domain-model invariants: kind-gating and the absorbed-id self-loop.

The one Card type drives insight-vs-exemplar behavior off ``kind`` alone, so the
kind gate (exemplar fields only on ``kind=program``) and the absorbed-id
self-loop guard are the model's load-bearing invariants.
"""

from __future__ import annotations

import pytest

from gigaevo.memory.cards import Card, CardKind, ContextualGain, DecisionContext


def test_insight_card_rejects_exemplar_fields() -> None:
    for field in (
        {"code": "x = 1"},
        {"program_id": "p1"},
        {"fitness": 0.5},
    ):
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


def test_program_card_accepts_metadata_without_code() -> None:
    card = Card(
        id="program-1",
        kind=CardKind.PROGRAM,
        program_id="p1",
        fitness=0.5,
    )
    assert card.code == ""


def test_card_cannot_absorb_its_own_id() -> None:
    with pytest.raises(ValueError):
        Card(id="mem-1", absorbed_ids=("mem-2", "mem-1"))


def test_card_absorbs_other_ids() -> None:
    card = Card(id="mem-1", absorbed_ids=("mem-2", "mem-3"))
    assert card.absorbed_ids == ("mem-2", "mem-3")


def test_legacy_card_and_gain_event_default_task_keys_to_empty() -> None:
    legacy = {
        "id": "mem-legacy",
        "gain_events": [
            {
                "context": {
                    "parent_metrics": {"fitness": 0.5},
                    "parent_id": "parent-1",
                },
                "gain": 0.2,
            }
        ],
    }

    card = Card.model_validate(legacy)
    event = ContextualGain.model_validate(legacy["gain_events"][0])

    assert card.task_key == ""
    assert card.gain_events[0].context.task_key == ""
    assert event.context.task_key == ""


def test_task_keys_survive_model_json_round_trip() -> None:
    card = Card(
        id="mem-task",
        task_key="heilbronn",
        gain_events=(
            ContextualGain(
                context=DecisionContext(task_key="heilbronn", parent_id="parent-1"),
                gain=0.2,
            ),
        ),
    )

    restored = Card.model_validate_json(card.model_dump_json())

    assert restored.task_key == "heilbronn"
    assert restored.gain_events[0].context.task_key == "heilbronn"
