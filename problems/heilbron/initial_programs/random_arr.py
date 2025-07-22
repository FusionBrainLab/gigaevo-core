import random

from helper import get_unit_triangle
import numpy as np

np.random.seed(42)
random.seed(42)


def entrypoint() -> np.ndarray:
    tri = get_unit_triangle()
    A, B, C = tri
    points = []
    for _ in range(11):
        r1 = np.random.rand()
        r2 = np.random.rand()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        P = (1 - r1 - r2) * A + r1 * B + r2 * C
        points.append(P)
    return np.array(points)
