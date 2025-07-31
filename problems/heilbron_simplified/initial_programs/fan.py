import random

from helper import get_unit_triangle, get_smallest_triangle_area, is_inside_triangle
import numpy as np

np.random.seed(42)
random.seed(42)


def entrypoint() -> np.ndarray:
    tri = get_unit_triangle()
    A, B, C = tri
    center = (A + B + C) / 3
    points = [center]
    for i in range(10):
        t = i / 10
        edge = A * (1 - t) + B * t if i % 2 == 0 else B * (1 - t) + C * t
        mid = (center + edge) / 2
        points.append(mid)
    return np.array(points)
