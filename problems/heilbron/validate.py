import itertools

from helper import get_unit_triangle
import numpy as np
from scipy.spatial.distance import pdist


def validate(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)

    # --- Input shape checks ---
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"Invalid shape for coordinates: expected (n, 2), got {coordinates.shape}"
        )
    if coordinates.shape[0] != 11:
        raise ValueError(f"Expected 11 points, got {coordinates.shape[0]}")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Some coordinates are NaN or infinite.")

    # --- Construct unit-area equilateral triangle ---
    A, B, C = get_unit_triangle()

    # --- Barycentric coordinate check (vectorized) ---
    def is_inside_triangle(points, a, b, c):
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

    if not is_inside_triangle(coordinates, A, B, C):
        raise ValueError("Some coordinates are outside the triangle.")

    dists = pdist(coordinates)
    min_dist = np.min(dists)

    if min_dist < 1e-6:
        raise ValueError("Some points are too close or overlapping.")

    # --- Vectorized triangle area computation for all (n choose 3) triplets ---
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

    return {
        "fitness": float(min_area),
        "is_valid": 1,
    }
