"""Fallback improver: small random jitter with greedy accept.

Used as cold-start opponent for Pop A when Pop B archive is empty.
"""

from helper import get_smallest_triangle_area, get_unit_triangle, is_inside_triangle
import numpy as np

np.random.seed(42)


def entrypoint():
    A, B, C = get_unit_triangle()

    def improve(points: np.ndarray) -> np.ndarray:
        best = points.copy()
        best_score = get_smallest_triangle_area(best)
        for _ in range(30):
            candidate = best.copy()
            idx = np.random.randint(0, 11)
            candidate[idx] += np.random.normal(0, 0.01, size=2)
            if not is_inside_triangle(candidate, A, B, C):
                continue
            score = get_smallest_triangle_area(candidate)
            if score > best_score:
                best = candidate
                best_score = score
        return best

    return improve
