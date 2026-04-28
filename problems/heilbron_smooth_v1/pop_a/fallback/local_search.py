"""Fallback improver: scipy-based local optimization.

Used as cold-start opponent for Pop A when Pop B archive is empty.
More powerful than jitter -- uses gradient-free optimization (Nelder-Mead).
"""

from helper import get_smallest_triangle_area, get_unit_triangle, is_inside_triangle
import numpy as np
from scipy.optimize import minimize

np.random.seed(42)


def entrypoint():
    A, B, C = get_unit_triangle()

    def improve(points: np.ndarray) -> np.ndarray:
        x0 = points.flatten()

        def objective(x):
            pts = x.reshape(11, 2)
            if not is_inside_triangle(pts, A, B, C):
                return 1e6  # penalty for leaving triangle
            return -get_smallest_triangle_area(pts)

        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-8},
        )
        improved = result.x.reshape(11, 2)
        if not is_inside_triangle(improved, A, B, C):
            return points  # fallback to original if optimization escaped
        return improved

    return improve
