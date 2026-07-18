"""Write-side eviction surface over the card bank.

The write path must not import the read system. The ``Evictor`` protocol and
``foreign_retention_veto`` are the eviction surface consumed by the admission
gate's periodic sweep. ``NullEvictor`` disables write-side eviction (the
``memory=v2`` default), the bank-maintenance twin of ``memory=none`` on the read
side. A future harm evictor returns through the same ``Evictor`` seam.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from gigaevo.memory.cards import Card, ContextualGain
from gigaevo.memory.context.evidence import sign_help_counts


@runtime_checkable
class Evictor(Protocol):
    def should_evict(self, card: Card) -> bool: ...

    def eviction_reason(self, card: Card) -> str: ...

    def sweep(self, cards: Sequence[Card]) -> list[str]: ...


def foreign_retention_veto(
    card: Card, *, task_key: str, min_effective_events: float
) -> str | None:
    """Return why foreign hard-sign evidence vetoes deletion, if it does."""
    if not task_key or min_effective_events <= 0.0:
        return None
    by_task: dict[str, list[ContextualGain]] = {}
    for event in card.gain_events:
        foreign_task = event.context.task_key
        if foreign_task and foreign_task != task_key:
            by_task.setdefault(foreign_task, []).append(event)
    for foreign_task in sorted(by_task):
        help_mass, total_mass = sign_help_counts(by_task[foreign_task])
        if total_mass >= min_effective_events and help_mass > 0.5 * total_mass:
            return (
                f"deletion vetoed by foreign task {foreign_task} "
                f"help {help_mass:.6g}/{total_mass:.6g}"
            )
    return None


class NullEvictor:
    """No-op evictor: never evicts. Runs the write path with eviction sweeps
    disabled, the bank-maintenance twin of ``memory=none`` on the read side."""

    def should_evict(self, card: Card) -> bool:
        return False

    def eviction_reason(self, card: Card) -> str:
        del card
        return ""

    def sweep(self, cards: Sequence[Card]) -> list[str]:
        return []
