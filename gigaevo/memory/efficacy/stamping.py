"""Stamping: attach use-attributed gain events onto cards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from gigaevo.memory.context import ContextualGain
from gigaevo.memory.shared_memory.models import MemoryCard

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
        A merge/consolidation survivor also folds in the events the pool still
        attributes to its ``absorbed_ids`` — children frozen with a since-merged
        card id credit that id, which no longer exists in the bank, so without the
        re-alias their attribution would orphan on the deleted id.

        Multiplicity is the trial count the harm gate reads (intro_events): every
        invalid child of one base parent emits a value-identical forced-harm event,
        and those are distinct trials that must all survive — on the own-id list and
        on a folded absorbed-id list alike. The absorbed fold still has to drop the
        ONE event a single child contributes when it credited both the survivor and
        an absorbed id, but that is the same event *object*: ``card_gain_events_from_programs``
        binds one ``ContextualGain`` per child and appends it to every credited id's
        list, so identity (not value) is the trial discriminator. Dedup by object
        identity keeps distinct value-equal trials and drops only the shared one.
        This relies on receiving one sweep's freshly built pool, which is the sole
        caller's contract (CardStatsUpdater.update).
        """
        folded: list[ContextualGain] = list(gain_events.get(card.id.strip()) or [])
        if isinstance(card, MemoryCard) and card.absorbed_ids:
            seen = {id(event) for event in folded}
            for aid in card.absorbed_ids:
                for event in gain_events.get(aid.strip()) or []:
                    if id(event) not in seen:
                        seen.add(id(event))
                        folded.append(event)
        return card.model_copy(update={"gain_events": folded or None})
