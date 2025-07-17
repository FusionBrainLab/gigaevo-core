from helper import get_unit_triangle
import numpy as np
import random

np.random.seed(42)
random.seed(42)

def entrypoint() -> np.ndarray:
    tri = get_unit_triangle()
    A, B, C = tri
    points = []
    for i in range(11):
        t = i / 10
        P1 = (1 - t) * A + t * B
        P2 = (1 - t) * B + t * C
        blend = (P1 + P2) / 2
        points.append(blend)
    return np.array(points)