"""Per-card EV magnitude read for the EV-bid auction.

``BetaBinomialReputation.card_magnitude`` resolves the card's expected gain
(``IntroGain_best_median``) from its raw gain events at read time, so the EV
auction can bid ``theta x magnitude``. Cards with no events are cold (``None``)
and fall back to the auction's optimistic prior.
"""

from __future__ import annotations

from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.shared_memory.models import (
    MemoryCard,
    MemoryCardExplanation,
)


def _card(gain_events: list[ContextualGain] | None) -> MemoryCard:
    return MemoryCard(
        id="m1",
        description="d",
        keywords=[],
        gain_events=gain_events,
        explanation=MemoryCardExplanation(summary=""),
    )


def _events(*gains: float) -> list[ContextualGain]:
    return [
        ContextualGain(
            context=DecisionContext(parent_metrics={"min_area": 0.5}), gain=g
        )
        for g in gains
    ]


class TestCardMagnitude:
    def test_reads_resolved_median_gain(self) -> None:
        assert BetaBinomialReputation().card_magnitude(_card(_events(0.0123))) == 0.0123

    def test_negative_magnitude_passes_through(self) -> None:
        assert BetaBinomialReputation().card_magnitude(_card(_events(-0.05))) == -0.05

    def test_no_events_is_cold_none(self) -> None:
        assert BetaBinomialReputation().card_magnitude(_card(None)) is None

    def test_empty_events_is_cold_none(self) -> None:
        assert BetaBinomialReputation().card_magnitude(_card([])) is None
