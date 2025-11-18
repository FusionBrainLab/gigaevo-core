import random

from helper import get_unit_triangle
import numpy as np

np.random.seed(42)
random.seed(42)


def entrypoint() -> np.ndarray:
    tri = get_unit_triangle()
    A, B, C = tri
    points = []
    rows = 5
    count = 0
    for row in range(rows):
        num_points = rows - row
        v = (row + 0.5) / rows
        for i in range(num_points):
            if count >= 11:
                break
            u = (i + 0.5) / num_points * (1 - v)
            P = (1 - u - v) * A + u * B + v * C
            perturbation = np.random.uniform(-0.001, 0.001, size=2)
            P = P + perturbation
            points.append(P)
            count += 1
        if count >= 11:
            break
    return np.array(points)
