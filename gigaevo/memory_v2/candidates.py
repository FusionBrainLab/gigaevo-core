"""Deterministic candidate generation for the v2 posterior policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from gigaevo.memory.cards import Card
from gigaevo.memory.read.exclusion import (
    CardExcluder,
    NullExcluder,
    is_card_excluded,
)
from gigaevo.memory.storage.base import MemoryStore
from gigaevo.programs.program import Program


@runtime_checkable
class CandidateSource(Protocol):
    def lineage_registry(self, *, task_key: str) -> tuple[Card, ...]: ...

    def candidates(
        self,
        program: Program,
        *,
        task_key: str,
        task_description: str,
    ) -> tuple[Card, ...]: ...

    def candidate_snapshot(
        self,
        program: Program,
        *,
        task_key: str,
        task_description: str,
    ) -> tuple[tuple[Card, ...], tuple[Card, ...]]: ...


class WholeBankCandidateSource:
    """Return the eligible bank in stable id order, without retrieval features."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        excluder: CardExcluder | None = None,
        allow_cross_task: bool = True,
        allowed_kinds: Sequence[str] = ("insight", "program"),
        max_candidates: int = 0,
    ) -> None:
        if max_candidates < 0:
            raise ValueError("max_candidates must be non-negative")
        self.store = store
        self.excluder = excluder if excluder is not None else NullExcluder()
        self.allow_cross_task = allow_cross_task
        self.allowed_kinds = frozenset(str(kind) for kind in allowed_kinds)
        self.max_candidates = max_candidates

    def candidates(
        self,
        program: Program,
        *,
        task_key: str,
        task_description: str,
    ) -> tuple[Card, ...]:
        return self.candidate_snapshot(
            program,
            task_key=task_key,
            task_description=task_description,
        )[1]

    def candidate_snapshot(
        self,
        program: Program,
        *,
        task_key: str,
        task_description: str,
    ) -> tuple[tuple[Card, ...], tuple[Card, ...]]:
        snapshot = self.store.snapshot()
        registry = self._registry_from_snapshot(snapshot, task_key=task_key)
        excluded = self.excluder.exclude_for(program)
        cards = [card for card in registry if not is_card_excluded(card, excluded)]
        del task_description
        cards.sort(key=lambda card: card.id)
        if self.max_candidates:
            cards = cards[: self.max_candidates]
        return registry, tuple(cards)

    def lineage_registry(self, *, task_key: str) -> tuple[Card, ...]:
        """Return a stable store-wide schema before contextual exclusion/caps."""

        return self._registry_from_snapshot(self.store.snapshot(), task_key=task_key)

    def _registry_from_snapshot(
        self, snapshot: Sequence[Card], *, task_key: str
    ) -> tuple[Card, ...]:
        cards = [
            card
            for card in snapshot
            if card.description.strip()
            and str(card.kind) in self.allowed_kinds
            and (
                self.allow_cross_task
                or not card.task_key
                or not task_key
                or card.task_key == task_key
            )
        ]
        return tuple(sorted(cards, key=lambda card: card.id))
