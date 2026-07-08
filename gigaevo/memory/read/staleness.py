"""Bank-cycle staleness: the single evidence-ageing mechanism.

Staleness ``s`` of a card = the number of gain events stamped anywhere in the
bank strictly newer than the card's own latest event — an internal rank count,
no wall clock. The half-life is ``half_life_cycles`` bank sizes (``H = len(bank)
* half_life_cycles`` — one cycle ~= every card earning one event), so the
discount ``w = 2**(-s / H)`` is self-normalizing: it scales with how fast the
bank is actually being exercised, with no absolute constants.

Two consumers share this one function: :class:`~gigaevo.memory.read.decay.
DecayingReputation` discounts a card's Beta posterior toward the cold prior by
``w``; the bootstrap auction uses the same ``w`` as the per-event resample
weight, so stale known-card deltas fade toward neutral zero. One mechanism, two
readers.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from gigaevo.memory.cards import Card, ContextualGain


def stamp(value: datetime | None) -> datetime | None:
    """UTC-normalize a possibly-naive event timestamp (bank stamps are UTC)."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def latest_event_stamp(card: Card) -> datetime | None:
    """The card's newest gain-event timestamp, or ``None`` when unstamped."""
    return latest_stamp(card.gain_events)


def latest_stamp(events: Sequence[ContextualGain]) -> datetime | None:
    """Newest timestamp in an evidence subset, or ``None`` when unstamped."""
    stamps = [
        stamped
        for event in events
        if (stamped := stamp(event.context.timestamp)) is not None
    ]
    return max(stamps) if stamps else None


def bank_cycle_weight(
    card: Card,
    bank: Sequence[Card],
    half_life_cycles: float,
    *,
    reference_events: Sequence[ContextualGain] | None = None,
) -> float:
    """Evidence discount ``w = 2**(-s / H)`` for a card at read time.

    ``s`` counts bank gain events strictly newer than the card's latest event;
    ``H = len(bank) * half_life_cycles``. Returns ``1.0`` (no discount) for a
    card with no stamped events or against an empty/degenerate bank.
    """
    latest = (
        latest_stamp(reference_events)
        if reference_events is not None
        else latest_event_stamp(card)
    )
    if latest is None:
        return 1.0
    half_life = len(bank) * half_life_cycles
    if half_life <= 0:
        return 1.0
    staleness = sum(
        1
        for banked in bank
        for event in banked.gain_events
        if (stamped := stamp(event.context.timestamp)) is not None and stamped > latest
    )
    return float(2.0 ** (-staleness / half_life))
