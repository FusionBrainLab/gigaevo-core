import math
from typing import List

from src.programs.program import Program


def extract_fitness_values(
    program: Program,
    fitness_keys: list[str],
    fitness_key_higher_is_better: dict[str, bool],
) -> list[float]:

    assert set(fitness_keys) == set(
        fitness_key_higher_is_better.keys()
    ), "All fitness keys must be present in the fitness_key_higher_is_better dict"

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
