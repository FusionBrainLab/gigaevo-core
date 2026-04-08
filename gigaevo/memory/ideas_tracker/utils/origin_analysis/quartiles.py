"""Generation quartile boundary computation."""
from __future__ import annotations


def generation_quantile_bounds(
    gens: list[int], qs: tuple[float, float, float] = (0.25, 0.50, 0.75)
) -> tuple[float, float, float]:
    gs = sorted(gens)
    if not gs:
        raise ValueError("No generations available.")

    def qval(q: float) -> float:
        idx = int(round(q * (len(gs) - 1)))
        idx = max(0, min(len(gs) - 1, idx))
        return float(gs[idx])

    return qval(qs[0]), qval(qs[1]), qval(qs[2])


def generation_range_bounds(gens: list[int]) -> tuple[float, float, float]:
    gmin, gmax = min(gens), max(gens)
    span = (gmax - gmin) + 1
    b1 = gmin + 0.25 * span
    b2 = gmin + 0.50 * span
    b3 = gmin + 0.75 * span
    return b1, b2, b3


def generation_to_quartile(gen: int, b1: float, b2: float, b3: float) -> str:
    if gen < b1:
        return "Q1"
    if gen < b2:
        return "Q2"
    if gen < b3:
        return "Q3"
    return "Q4"
