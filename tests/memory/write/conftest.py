"""Fixtures for the write layer: card/program factories and an in-memory store."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

import pytest

from gigaevo.exceptions import MergeAborted
from gigaevo.memory.cards import Card, CardKind, ContextualGain, DecisionContext
from gigaevo.memory.storage.base import (
    MemoryStore,
    MergeRetireResult,
    ResearchRequest,
    ResearchResult,
    ScoredCard,
)
from gigaevo.programs.metrics.context import (
    VALIDITY_KEY,
    MetricsContext,
    MetricSpec,
)
from gigaevo.programs.program import Lineage, Program


class FakeStore(MemoryStore):
    """Dict-backed MemoryStore honoring the id-minting and merge contracts."""

    def __init__(self) -> None:
        self._cards: dict[str, Card] = {}
        self._lock = RLock()
        self._minted = 0
        self.hits: list[ScoredCard] = []
        self.fail_merges = False
        self.saved_ids: list[str] = []
        self.deleted_ids: list[str] = []

    @property
    def is_ready(self) -> bool:
        return True

    def save(self, card: Card) -> str:
        with self._lock:
            if not card.id:
                self._minted += 1
                card = card.model_copy(update={"id": f"minted-{self._minted:04d}"})
            self._cards[card.id] = card
            self.saved_ids.append(card.id)
            return card.id

    def update(
        self, card_id: str, transform: Callable[[Card], Card | None]
    ) -> Card | None:
        with self._lock:
            current = self._cards.get(card_id)
            if current is None:
                return None
            replacement = transform(current)
            if replacement is not None and replacement.id != card_id:
                raise ValueError("atomic card update cannot change the card id")
            if replacement is None:
                self._cards.pop(card_id)
                self.deleted_ids.append(card_id)
                return current
            if replacement != current:
                self._cards[card_id] = replacement
                self.saved_ids.append(card_id)
            return replacement

    def get(self, card_id: str) -> Card | None:
        with self._lock:
            return self._cards.get(card_id)

    def delete(self, card_id: str) -> bool:
        with self._lock:
            self.deleted_ids.append(card_id)
            return self._cards.pop(card_id, None) is not None

    def snapshot(self) -> tuple[Card, ...]:
        with self._lock:
            return tuple(self._cards[cid] for cid in sorted(self._cards))

    def merge_retire(
        self,
        target_id: str,
        partner_id: str,
        fold: Callable[[Card, Card | None], Card | None],
    ) -> MergeRetireResult:
        with self._lock:
            target = self._cards.get(target_id)
            partner = self._cards.get(partner_id) if partner_id else None
            if target is None:
                return MergeRetireResult(outcome="target_missing")
            if self.fail_merges or partner_id == target_id:
                return MergeRetireResult(outcome="aborted")
            if partner is not None and target_id in partner.absorbed_ids:
                return MergeRetireResult(outcome="aborted")
            try:
                survivor = fold(target, partner)
            except MergeAborted:
                return MergeRetireResult(outcome="aborted")
            if survivor is not None and survivor.id != target_id:
                raise ValueError("atomic merge survivor must keep the target id")
            if survivor is not None and survivor.id in survivor.absorbed_ids:
                return MergeRetireResult(outcome="aborted")
            if survivor is None:
                self._cards.pop(target_id)
                self.deleted_ids.append(target_id)
                outcome = "retired"
            else:
                self._cards[target_id] = survivor
                self.saved_ids.append(target_id)
                outcome = "merged"
            if partner is not None:
                self._cards.pop(partner_id)
                self.deleted_ids.append(partner_id)
            return MergeRetireResult(outcome=outcome)

    def nearest(
        self, text: str, k: int, kind: CardKind | None = None
    ) -> list[ScoredCard]:
        hits = [h for h in self.hits if kind is None or h.card.kind is kind]
        return hits[:k]

    async def research(self, request: ResearchRequest) -> ResearchResult:
        return ResearchResult()

    def rebuild(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def make_card():
    counter = iter(range(10_000))

    def _make_card(**overrides) -> Card:
        n = next(counter)
        params = {
            "id": f"mem-test{n:04d}",
            "kind": CardKind.INSIGHT,
            "description": f"idea-{n} exploits problem structure",
            "explanation_summary": f"works because of invariant-{n}",
        }
        params.update(overrides)
        return Card(**params)

    return _make_card


@pytest.fixture
def make_event():
    def _make_event(
        gain: float,
        *,
        invalid: bool = False,
        founding: bool = False,
        unused: bool = False,
        metrics: dict[str, float] | None = None,
        parent_id: str = "",
        task_key: str = "",
    ) -> ContextualGain:
        return ContextualGain(
            context=DecisionContext(
                task_key=task_key,
                parent_metrics=metrics or {},
                parent_id=parent_id,
            ),
            gain=gain,
            invalid=invalid,
            founding=founding,
            unused=unused,
        )

    return _make_event


@pytest.fixture
def metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="fitness", higher_is_better=True, is_primary=True
            )
        }
    )


@pytest.fixture
def make_program():
    counter = iter(range(10_000))

    def _make_program(
        *,
        fitness: float | None = 0.5,
        valid: float | None = 1.0,
        parents: list[str] | None = None,
        metadata: dict | None = None,
        **overrides,
    ) -> Program:
        n = next(counter)
        metrics: dict[str, float] = {}
        if valid is not None:
            metrics[VALIDITY_KEY] = valid
        if fitness is not None:
            metrics["fitness"] = fitness
        params = {
            "code": f"x = {n}",
            "metrics": metrics,
            "metadata": metadata or {},
            "lineage": Lineage(parents=parents or [], generation=1, mutation=None),
        }
        params.update(overrides)
        return Program(**params)

    return _make_program
