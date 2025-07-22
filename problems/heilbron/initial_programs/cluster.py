import random

from helper import get_unit_triangle
import numpy as np

np.random.seed(42)
random.seed(42)


def entrypoint() -> np.ndarray:
    tri = get_unit_triangle()
    A, B, C = tri
    center = (A + B + C) / 3
    points = [center]
    radius = 0.2 * np.linalg.norm(B - A)
    for i in range(10):
        angle = 2 * np.pi * i / 10
        offset = np.array([np.cos(angle), np.sin(angle)]) * radius
        P = center + offset
        # Project back into triangle using barycentric clipping if needed
        alpha, beta, gamma = np.linalg.solve(
            np.column_stack([A - C, B - C]), (P - C)
        ).tolist() + [1]
        if any(v < 0 for v in [alpha, beta, 1 - alpha - beta]):
            P = center  # fallback
        points.append(P)
    return np.array(points)
