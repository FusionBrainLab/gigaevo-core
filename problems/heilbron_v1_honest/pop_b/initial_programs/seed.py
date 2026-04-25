"""Seed improver: random perturbation with greedy accept.

Tries random jitters on each point, keeps changes that increase min_area.
Simple but effective baseline for the Improver population.
"""

from helper import get_smallest_triangle_area, get_unit_triangle, is_inside_triangle
import numpy as np

np.random.seed(42)


def entrypoint():
    """Return an improve(points) -> improved_points callable."""
    A, B, C = get_unit_triangle()

    def improve(points: np.ndarray) -> np.ndarray:
        best = points.copy()
        best_score = get_smallest_triangle_area(best)

        for _round in range(50):
            candidate = best.copy()
            # Pick a random point and perturb it
            idx = np.random.randint(0, 11)
            perturbation = np.random.normal(0, 0.02, size=2)
            candidate[idx] += perturbation

            # Check containment
            if not is_inside_triangle(candidate, A, B, C):
                continue

            score = get_smallest_triangle_area(candidate)
            if score > best_score:
                best = candidate
                best_score = score

        return best

    return improve
