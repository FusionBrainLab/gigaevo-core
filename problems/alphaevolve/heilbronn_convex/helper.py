import itertools
import numpy as np


def get_smallest_triangle_area(coordinates: np.ndarray) -> float:
    n = coordinates.shape[0]
    idx = np.array(list(itertools.combinations(range(n), 3)))
    pts = coordinates[idx]

    a = pts[:, 0, :]
    b = pts[:, 1, :]
    c = pts[:, 2, :]

    areas = 0.5 * np.abs(
        (
            a[:, 0] * (b[:, 1] - c[:, 1])
            + b[:, 0] * (c[:, 1] - a[:, 1])
            + c[:, 0] * (a[:, 1] - b[:, 1])
        )
    )

    min_area = np.min(areas)
    return min_area
