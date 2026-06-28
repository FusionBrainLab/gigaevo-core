"""The card's ``explanation_summary`` is a first-class retrieval channel.

A-MEM indexes every card field into its own Chroma collection, including
``memories_explanation_summary``. That channel sits dark unless the card model
carries an ``explanation_summary`` and every serialization seam on the write
path round-trips it: model -> ``model_dump`` (-> GAM pages.json -> Chroma),
raw-payload -> typed card, and card <-> API concept. These tests pin the field
through each seam so the channel is fed, not silently dropped.
"""

from __future__ import annotations

from gigaevo.memory.shared_memory.card_conversion import (
    card_to_concept_content,
    concept_to_card,
    normalize_memory_card,
)
from gigaevo.memory.shared_memory.models import MemoryCard, ProgramCard


def test_memory_card_serializes_explanation_summary() -> None:
    card = MemoryCard(id="m1", explanation_summary="why this lever escapes the trap")
    assert card.model_dump()["explanation_summary"] == "why this lever escapes the trap"


def test_program_card_serializes_explanation_summary() -> None:
    card = ProgramCard(id="p1", explanation_summary="why this exemplar scores well")
    assert card.model_dump()["explanation_summary"] == "why this exemplar scores well"


def test_normalize_preserves_memory_card_explanation_summary() -> None:
    card = normalize_memory_card(
        {"id": "m1", "description": "d", "explanation_summary": "condensed why"}
    )
    assert isinstance(card, MemoryCard)
    assert card.explanation_summary == "condensed why"


def test_normalize_preserves_program_card_explanation_summary() -> None:
    card = normalize_memory_card(
        {"id": "p1", "category": "program", "explanation_summary": "condensed why"}
    )
    assert isinstance(card, ProgramCard)
    assert card.explanation_summary == "condensed why"


def test_concept_content_includes_memory_card_explanation_summary() -> None:
    card = MemoryCard(id="m1", explanation_summary="condensed why")
    assert card_to_concept_content(card)["explanation_summary"] == "condensed why"


def test_concept_content_includes_program_card_explanation_summary() -> None:
    card = ProgramCard(id="p1", explanation_summary="condensed why")
    assert card_to_concept_content(card)["explanation_summary"] == "condensed why"


def test_concept_roundtrip_preserves_explanation_summary() -> None:
    card = MemoryCard(id="m1", description="d", explanation_summary="condensed why")
    roundtripped = concept_to_card(card_to_concept_content(card), fallback_id="fb")
    assert roundtripped.explanation_summary == "condensed why"


def test_program_card_concept_roundtrip_preserves_keywords_and_gain_events() -> None:
    # The card<->API concept seam must not drop ProgramCard's keywords/gain_events:
    # the MemoryCard branch carries them, so an API-backed round-trip of a program
    # exemplar must reconstruct them rather than blanking keywords/gain_events.
    from gigaevo.memory.context import ContextualGain, DecisionContext

    gain = ContextualGain(context=DecisionContext(parent_metrics={"f": 0.1}), gain=0.1)
    card = ProgramCard(
        id="p1",
        program_id="prog-1",
        description="exemplar does X",
        explanation_summary="why it scores",
        keywords=["alpha", "beta"],
        gain_events=[gain],
    )
    roundtripped = concept_to_card(card_to_concept_content(card), fallback_id="fb")
    assert isinstance(roundtripped, ProgramCard)
    assert roundtripped.description == "exemplar does X"
    assert roundtripped.explanation_summary == "why it scores"
    assert roundtripped.keywords == ["alpha", "beta"]
    assert roundtripped.gain_events == [gain]


def test_memory_card_concept_roundtrip_preserves_programs() -> None:
    # The card<->API concept seam must not drop MemoryCard.programs (provenance):
    # to_card() can reconstruct programs, so an API-backed round-trip of an idea
    # card must keep its provenance rather than blanking it on save/load.
    card = MemoryCard(
        id="m1",
        description="lever does X",
        explanation_summary="why it works",
        programs=["prog-1", "prog-2"],
        keywords=["alpha", "beta"],
    )
    roundtripped = concept_to_card(card_to_concept_content(card), fallback_id="fb")
    assert isinstance(roundtripped, MemoryCard)
    assert roundtripped.programs == ["prog-1", "prog-2"]
    assert roundtripped.keywords == ["alpha", "beta"]
    assert roundtripped.explanation_summary == "why it works"
