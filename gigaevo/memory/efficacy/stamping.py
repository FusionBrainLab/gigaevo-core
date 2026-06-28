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
        """Card with the current sweep's authoritative gain events attached.

        The full pool is authoritative each sweep: a credited card carries this
        sweep's events; an uncredited card has any stale events cleared to None.
        """
        events = gain_events.get(card.id.strip())
        return card.model_copy(update={"gain_events": list(events) if events else None})
