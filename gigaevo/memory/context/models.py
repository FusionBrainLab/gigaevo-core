"""Context partitioning for memory evidence and no-card baselines.

``DecisionContext`` stays a small serialized value object on card events.  These
models are the configurable policy layer that decides which events/outcomes are
comparable to a read decision: global for portable runs, or BD-cell local for
single-island MAP-Elites style runs.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import statistics
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.evolution.strategies.models import BehaviorSpace
from gigaevo.memory.cards import Card, ContextualGain, DecisionContext


@runtime_checkable
class ParentContextSource(Protocol):
    """Program-like parent fields needed to build a memory decision context."""

    @property
    def id(self) -> str: ...

    @property
    def metrics(self) -> dict[str, float]: ...


@runtime_checkable
class NoCardBaselineOutcome(Protocol):
    """Outcome fields needed by no-card baseline estimators."""

    @property
    def fitness(self) -> float | None: ...

    @property
    def invalid(self) -> bool: ...

    @property
    def base_selected_ids(self) -> Sequence[str]: ...

    @property
    def base_metrics(self) -> dict[str, float]: ...

    @property
    def base_fitness(self) -> float | None: ...

    @property
    def no_card_control(self) -> bool: ...


class ContextKey(BaseModel):
    """Stable, serializable identity of the context bucket a read belongs to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = Field(description="Context model namespace, e.g. global or bd_cell.")
    parts: tuple[str, ...] = Field(
        default=(), description="Hashable bucket coordinates within the namespace."
    )

    def label(self) -> str:
        return self.kind if not self.parts else f"{self.kind}:{'/'.join(self.parts)}"


class _ConstantNoCardBaseline(BaseModel):
    model_config = ConfigDict(frozen=True)

    baseline: float = 0.0
    has_evidence: bool = False

    def baseline_for(self, outcome: NoCardBaselineOutcome) -> float:
        del outcome
        return self.baseline


class _CellNoCardBaseline(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    behavior_space: BehaviorSpace
    global_median: float = 0.0
    by_cell: dict[tuple[int, ...], float] = Field(default_factory=dict)
    has_evidence: bool = False

    def baseline_for(self, outcome: NoCardBaselineOutcome) -> float:
        cell = _cell_in(self.behavior_space, outcome.base_metrics)
        if cell is not None and cell in self.by_cell:
            return self.by_cell[cell]
        return self.global_median


@runtime_checkable
class MemoryContextModel(Protocol):
    """Configurable context policy for memory evidence aggregation."""

    def read_context(
        self, parents: Sequence[ParentContextSource]
    ) -> DecisionContext | None: ...

    def key_for(self, context: DecisionContext | None = None) -> ContextKey: ...

    def evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]: ...

    def local_evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]: ...

    def fit_no_card_baseline(
        self, outcomes: Sequence[NoCardBaselineOutcome], *, higher_is_better: bool
    ) -> Any: ...


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _clean_ids(ids: Sequence[str]) -> set[str]:
    return {c.strip() for c in ids if c.strip()}


def _outcome_selected_ids(outcome: NoCardBaselineOutcome) -> set[str]:
    return _clean_ids(tuple(outcome.base_selected_ids or ()))


def _outcome_no_card_control(outcome: NoCardBaselineOutcome) -> bool:
    return bool(outcome.no_card_control)


def _oriented_delta(
    outcome: NoCardBaselineOutcome, higher_is_better: bool
) -> float | None:
    fitness = outcome.fitness
    base_fitness = outcome.base_fitness
    if fitness is None or base_fitness is None:
        return None
    return (
        float(fitness) - float(base_fitness)
        if higher_is_better
        else float(base_fitness) - float(fitness)
    )


def _no_card_cohort(
    outcomes: Sequence[NoCardBaselineOutcome],
) -> list[NoCardBaselineOutcome]:
    no_card = [
        outcome
        for outcome in outcomes
        if outcome.base_fitness is not None and not _outcome_selected_ids(outcome)
    ]
    controls = [outcome for outcome in no_card if _outcome_no_card_control(outcome)]
    return controls if controls else no_card


def _no_card_deltas(
    outcomes: Sequence[NoCardBaselineOutcome], higher_is_better: bool
) -> list[float]:
    deltas: list[float] = []
    for outcome in _no_card_cohort(outcomes):
        if outcome.invalid:
            deltas.append(0.0)
            continue
        delta = _oriented_delta(outcome, higher_is_better)
        if delta is not None and math.isfinite(delta):
            deltas.append(delta)
    return deltas


def _read_context_from_parents(
    parents: Sequence[ParentContextSource],
) -> DecisionContext | None:
    if not parents:
        return None
    parent = parents[0]
    return DecisionContext(
        parent_metrics=dict(parent.metrics or {}),
        parent_id=str(parent.id or ""),
    )


def _event_weight(event: ContextualGain) -> float:
    attr = event.attribution
    if attr is not None and attr.credit_weight is not None:
        weight = float(attr.credit_weight)
    elif event.founding:
        weight = 0.0
    else:
        weight = 1.0
    return weight if math.isfinite(weight) and weight > 0.0 else 0.0


def _has_non_founding_support(events: Sequence[ContextualGain]) -> bool:
    for event in events:
        if event.founding or _event_weight(event) <= 0.0:
            continue
        if event.invalid or event.unused:
            return True
        if event.gain is not None and math.isfinite(float(event.gain)):
            return True
    return False


def _cell_in(space: BehaviorSpace, metrics: dict[str, float]) -> tuple[int, ...] | None:
    for key in space.behavior_keys:
        value = metrics.get(key)
        if value is None or not math.isfinite(value):
            return None
    return space.get_cell(metrics)


class GlobalMemoryContext(BaseModel):
    """Portable context model: every event belongs to one global bucket."""

    model_config = ConfigDict(frozen=True)

    def read_context(
        self, parents: Sequence[ParentContextSource]
    ) -> DecisionContext | None:
        return _read_context_from_parents(parents)

    def key_for(self, context: DecisionContext | None = None) -> ContextKey:
        del context
        return ContextKey(kind="global")

    def evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]:
        del context
        return tuple(card.gain_events)

    def local_evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]:
        return self.evidence_events(card, context)

    def fit_no_card_baseline(
        self, outcomes: Sequence[NoCardBaselineOutcome], *, higher_is_better: bool
    ) -> Any:
        deltas = _no_card_deltas(outcomes, higher_is_better)
        return _ConstantNoCardBaseline(
            baseline=_median(deltas), has_evidence=bool(deltas)
        )


class BDCellMemoryContext(BaseModel):
    """BD-cell local context model with global fallback.

    Cell ids are deliberately recomputed at read/fit time under a copied
    behavior space.  ``ContextualGain`` remains free of stored cell ids, so a
    dynamic behavior-space reindex never leaves stale buckets serialized on
    cards.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    behavior_space: BehaviorSpace = Field(
        description="Shared behavior-space tessellation for this run."
    )
    fallback: GlobalMemoryContext = Field(default_factory=GlobalMemoryContext)

    def read_context(
        self, parents: Sequence[ParentContextSource]
    ) -> DecisionContext | None:
        return self.fallback.read_context(parents)

    def key_for(self, context: DecisionContext | None = None) -> ContextKey:
        if context is None:
            return self.fallback.key_for(context)
        space = self.behavior_space.model_copy(deep=True)
        cell = _cell_in(space, context.parent_metrics)
        if cell is None:
            return self.fallback.key_for(context)
        return ContextKey(kind="bd_cell", parts=tuple(str(part) for part in cell))

    def evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]:
        local = self.local_evidence_events(card, context)
        if local:
            return local
        return self.fallback.evidence_events(card, context)

    def local_evidence_events(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[ContextualGain, ...]:
        if context is None or not card.gain_events:
            return self.fallback.evidence_events(card, context)
        space = self.behavior_space.model_copy(deep=True)
        parent_cell = _cell_in(space, context.parent_metrics)
        if parent_cell is None:
            return ()
        in_cell = tuple(
            event
            for event in card.gain_events
            if _cell_in(space, event.context.parent_metrics) == parent_cell
        )
        return in_cell if _has_non_founding_support(in_cell) else ()

    def fit_no_card_baseline(
        self, outcomes: Sequence[NoCardBaselineOutcome], *, higher_is_better: bool
    ) -> Any:
        space = self.behavior_space.model_copy(deep=True)
        global_deltas: list[float] = []
        deltas_by_cell: dict[tuple[int, ...], list[float]] = {}
        for outcome in _no_card_cohort(outcomes):
            delta: float | None
            if outcome.invalid:
                delta = 0.0
            else:
                delta = _oriented_delta(outcome, higher_is_better)
            if delta is None or not math.isfinite(delta):
                continue
            global_deltas.append(delta)
            if (cell := _cell_in(space, outcome.base_metrics)) is not None:
                deltas_by_cell.setdefault(cell, []).append(delta)
        return _CellNoCardBaseline(
            behavior_space=space,
            global_median=_median(global_deltas),
            by_cell={cell: _median(deltas) for cell, deltas in deltas_by_cell.items()},
            has_evidence=bool(global_deltas),
        )
