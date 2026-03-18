import random
from typing import TypeVar

from loguru import logger

from gigaevo.programs.program import Program

_T = TypeVar("_T")


def weighted_sample_without_replacement[T](
    items: list[T],
    weights: list[float],
    k: int,
) -> list[T]:
    """Select *k* items by weighted sampling without replacement.

    When all remaining weights are zero, the remaining selections fall back
    to uniform random sampling.

    Parameters
    ----------
    items:
        Population to sample from.
    weights:
        Non-negative sampling weights aligned with *items*.
    k:
        Number of items to select (clamped to ``len(items)``).

    Returns
    -------
    list[_T]
        Selected items in order of selection.
    """
    k = min(k, len(items))
    selected: list[T] = []
    remaining_items = list(items)
    remaining_weights = list(weights)

    for _ in range(k):
        if not remaining_items:
            break

        total_weight = sum(remaining_weights)
        if total_weight == 0:
            logger.warning(
                "weighted_sample_without_replacement: all remaining weights "
                "are zero; falling back to uniform sampling "
                "(remaining={}, already_selected={}, requested_total={})",
                len(remaining_items),
                len(selected),
                k,
            )
            selected.extend(random.sample(remaining_items, k - len(selected)))
            break

        chosen = random.choices(remaining_items, weights=remaining_weights, k=1)[0]
        selected.append(chosen)

        idx = remaining_items.index(chosen)
        remaining_items.pop(idx)
        remaining_weights.pop(idx)

    return selected


def extract_fitness_values(
    program: Program,
    fitness_keys: list[str],
    fitness_key_higher_is_better: dict[str, bool],
) -> list[float]:
    assert set(fitness_keys) == set(fitness_key_higher_is_better.keys()), (
        "All fitness keys must be present in the fitness_key_higher_is_better dict"
    )

    values = []
    for key in fitness_keys:
        if key not in program.metrics:
            raise KeyError(f"Missing fitness key '{key}' in program metrics")

        value: float = program.metrics[key]
        values.append(value if fitness_key_higher_is_better[key] else -value)
    return values


def dominates(p: list[float], q: list[float]) -> bool:
    """Returns True if p Pareto-dominates q (i.e., p is ≥ in all and > in at least one)."""
    return all(p_i >= q_i for p_i, q_i in zip(p, q)) and any(
        p_i > q_i for p_i, q_i in zip(p, q)
    )
