from helper import get_unit_triangle, get_smallest_triangle_area, is_inside_triangle
import numpy as np
import random

np.random.seed(42)
random.seed(42)

def entrypoint() -> np.ndarray:
    tri = get_unit_triangle()
    A, B, C = tri
    points = []
    rows = 4
    count = 0
    for row in range(rows):
        num_points = row + 1
        for i in range(num_points):
            if count >= 11:
                return np.array(points)
            u = (i + 0.5) / num_points
            v = (row + 0.5) / rows
            if u + v > 1:
                continue
            P = (1 - u - v) * A + u * B + v * C
            points.append(P)
            count += 1
    return np.array(points)