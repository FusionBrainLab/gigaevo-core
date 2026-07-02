"""Harm eviction over the card bank.

The write path must not import the read system, but the harm verdict IS the
read side's injection posterior. ``CardScorer`` inverts that dependency: this
module declares the scoring surface it needs, ``read/reputation.py``'s
reputation models satisfy it structurally, and the integration config wires
one shared instance into both sides.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from loguru import logger

from gigaevo.memory.cards import Card, CardStatsBlock, DecisionContext
from gigaevo.memory.events import MemoryEvictionSweep, emit_memory_event


class CardScorer(Protocol):
    def card_stats(
        self, card: Card, context: DecisionContext | None = None
    ) -> CardStatsBlock | None: ...

    def is_confidently_harmful(self, block: CardStatsBlock | None) -> bool: ...


class Evictor(Protocol):
    def should_evict(self, card: Card) -> bool: ...

    def sweep(self, cards: Sequence[Card]) -> list[str]: ...


class HarmEvictor:
    """Evicts cards whose injection posterior is confidently harmful."""

    def __init__(self, scorer: CardScorer) -> None:
        self._scorer = scorer

    def should_evict(self, card: Card) -> bool:
        return self._scorer.is_confidently_harmful(self._scorer.card_stats(card))

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        evicted = [card.id for card in cards if self.should_evict(card)]
        if evicted:
            emit_memory_event(
                MemoryEvictionSweep(bank_count=len(cards), evicted_ids=tuple(evicted))
            )
            logger.info(
                "[Memory][Evictor] Sweep evicting {}/{} card(s) as confidently harmful: {}",
                len(evicted),
                len(cards),
                evicted,
            )
        return evicted


class NullEvictor:
    """No-op evictor: never evicts. Runs the write path with the harm sweep
    disabled, the bank-maintenance twin of ``memory=none`` on the read side."""

    def should_evict(self, card: Card) -> bool:
        return False

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        return []
