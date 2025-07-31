import itertools

import numpy as np


def get_unit_triangle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit_area_side = np.sqrt(4 / np.sqrt(3))  # scale for unit area
    height = np.sqrt(3) / 2 * unit_area_side
    A = np.array([0, 0])
    B = np.array([unit_area_side, 0])
    C = np.array([unit_area_side / 2, height])
    return A, B, C


def get_smallest_triangle_area(coordinates: np.ndarray) -> float:
    n = coordinates.shape[0]
    idx = np.array(list(itertools.combinations(range(n), 3)))
    pts = coordinates[idx]  # shape: (N, 3, 2)

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


def is_inside_triangle(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> bool:
    # Compute barycentric coordinates
    v0 = c - a
    v1 = b - a
    v2 = points - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)

    d20 = np.einsum("ij,j->i", v2, v0)
    d21 = np.einsum("ij,j->i", v2, v1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.all((u >= -1e-12) & (v >= -1e-12) & (w >= -1e-12))
