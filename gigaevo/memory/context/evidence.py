"""Shared evidence arithmetic used by memory read, write, and context layers.

These helpers define the one framework-wide meaning of an event's causal
weight, an oriented child-parent delta, and id hygiene; read/write modules must
not re-derive them locally.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import statistics

from gigaevo.memory.cards import ContextualGain


def event_weight(event: ContextualGain) -> float:
    attr = event.attribution
    if attr is not None and attr.credit_weight is not None:
        weight = float(attr.credit_weight)
    elif event.founding:
        weight = 0.0
    else:
        weight = 1.0
    return weight if math.isfinite(weight) and weight > 0.0 else 0.0


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
