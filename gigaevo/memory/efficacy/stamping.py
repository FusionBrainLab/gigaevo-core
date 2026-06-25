"""Stamping: attach use-attributed gain events onto cards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from gigaevo.memory.context import ContextualGain

if TYPE_CHECKING:
    from gigaevo.memory.shared_memory.models import AnyCard


class CardStatsStamper(BaseModel):
    """Single writer of card-side efficacy evidence: attaches the use-attributed
    gain events a card earned this sweep. The card stores only the raw events;
    reputation computes every statistic from them at read time."""

    model_config = ConfigDict(frozen=True)

    def stamp_gain_events(
        self, card: AnyCard, gain_events: dict[str, list[ContextualGain]]
    ) -> AnyCard:
        """Card with its use-attributed gain events attached; cards without
        events pass through unchanged."""
        events = gain_events.get(card.id.strip())
        if not events:
            return card
        return card.model_copy(update={"gain_events": list(events)})
