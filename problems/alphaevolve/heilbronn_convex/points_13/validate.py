import numpy as np
import itertools
from scipy.spatial import ConvexHull


def validate(points):
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Invalid shape: expected (13, 2), got {points.shape}")

    if points.shape[0] != 13:
        raise ValueError(f"Expected 13 points, got {points.shape[0]}")

    if not np.all(np.isfinite(points)):
        raise ValueError("Some coordinates are NaN or infinite")

    tol = 1e-10
    for i in range(13):
        for j in range(i + 1, 13):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < tol:
                raise ValueError(
                    f"Duplicate points: points {i} and {j} too close (distance: {dist:.2e})"
                )

    min_triangle_area = float("inf")
    for p1, p2, p3 in itertools.combinations(points, 3):
        area = abs(
            p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
        ) / 2
        if area < min_triangle_area:
            min_triangle_area = area

    if min_triangle_area < 1e-12:
        raise ValueError(
            f"Degenerate triangle detected: min_triangle_area = {min_triangle_area:.2e}"
        )

    convex_hull_area = ConvexHull(points).volume

    if convex_hull_area < 1e-10:
        raise ValueError("Convex hull area is too small")

    min_area_normalized = min_triangle_area / convex_hull_area

    return {"fitness": float(min_area_normalized), "is_valid": 1}
