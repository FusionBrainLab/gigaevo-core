"""Shared evidence arithmetic used by memory read, write, and context layers.

These helpers define the one framework-wide meaning of an event's causal
weight, an oriented child-parent delta, and id hygiene; read/write modules must
not re-derive them locally.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import statistics
from typing import Protocol

from scipy.stats import norm

from gigaevo.memory.cards import Card, ContextualGain, DecisionContext


class EffectiveSupportScorer(Protocol):
    def event_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]: ...

    def staleness_weights(
        self, card: Card, context: DecisionContext | None = None
    ) -> tuple[float, ...]: ...


def effective_support(
    scorer: EffectiveSupportScorer,
    card: Card,
    deltas: Sequence[float],
    context: DecisionContext | None,
) -> float:
    """Per-event aged effective support shared by read and write policy."""
    weights = scorer.event_weights(card, context)
    if len(weights) != len(deltas):
        raise ValueError(
            "event_weights must align with event_deltas: "
            f"{len(weights)} weights for {len(deltas)} deltas"
        )
    staleness = scorer.staleness_weights(card, context)
    if len(staleness) != len(deltas):
        raise ValueError(
            "staleness_weights must align with event_deltas: "
            f"{len(staleness)} weights for {len(deltas)} deltas"
        )
    terms = (float(credit) * float(age) for credit, age in zip(weights, staleness))
    return sum(max(0.0, term) for term in terms if math.isfinite(term))


def split_events_by_task(
    events: Sequence[ContextualGain], task_key: str
) -> tuple[tuple[ContextualGain, ...], tuple[ContextualGain, ...]]:
    """Partition events by exact origin-task equality, preserving order."""
    native: list[ContextualGain] = []
    foreign: list[ContextualGain] = []
    for event in events:
        (native if event.context.task_key == task_key else foreign).append(event)
    return tuple(native), tuple(foreign)


def event_weight(event: ContextualGain) -> float:
    attr = event.attribution
    if attr is not None and attr.credit_weight is not None:
        weight = float(attr.credit_weight)
    elif event.founding:
        weight = 0.0
    else:
        weight = 1.0
    return weight if math.isfinite(weight) and weight > 0.0 else 0.0


def harm_mass(gain: float, se: float | None, threshold: float) -> float:
    """Below-threshold probability for unknown, exact, or measured uncertainty."""
    if se is None:
        # As se -> infinity, the Gaussian tail tends to Phi(0): no sign information.
        return float(norm.cdf(0.0))
    if se <= 0.0:
        return 1.0 if gain < threshold else 0.0
    return float(norm.cdf((threshold - gain) / se))


def sign_help_counts(
    events: Sequence[ContextualGain],
    *,
    staleness_weights: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Hard-sign help and total mass from used exposure events."""
    ages = tuple(staleness_weights) if staleness_weights is not None else None
    if ages is not None and len(ages) != len(events):
        raise ValueError(
            "staleness_weights must align with events: "
            f"{len(ages)} weights for {len(events)} events"
        )
    help_mass = 0.0
    total_mass = 0.0
    for idx, event in enumerate(events):
        age = float(ages[idx]) if ages is not None else 1.0
        weight = event_weight(event) * age
        if event.founding or weight <= 0.0 or (event.unused and not event.invalid):
            continue
        if event.invalid:
            total_mass += weight
            continue
        gain = float(event.gain)
        if not math.isfinite(gain):
            continue
        if gain >= 0.0:
            help_mass += weight
        total_mass += weight
    return help_mass, total_mass


def oriented_delta(
    fitness: float | None, base_fitness: float | None, higher_is_better: bool
) -> float | None:
    if fitness is None or base_fitness is None:
        return None
    return (
        float(fitness) - float(base_fitness)
        if higher_is_better
        else float(base_fitness) - float(fitness)
    )


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def clean_ids(ids: Sequence[str]) -> set[str]:
    return {cid.strip() for cid in ids if cid.strip()}
