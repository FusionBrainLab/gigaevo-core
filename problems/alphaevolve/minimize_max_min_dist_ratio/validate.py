import numpy as np
from scipy.spatial.distance import pdist


def validate(points):
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Invalid shape: expected (16, 2), got {points.shape}")

    if points.shape[0] != 16:
        raise ValueError(f"Expected 16 points, got {points.shape[0]}")

    if not np.all(np.isfinite(points)):
        raise ValueError("Some coordinates are NaN or infinite.")

    distances = pdist(points)

    if len(distances) == 0:
        raise ValueError("Cannot compute distances for single point.")

    min_distance = np.min(distances)

    if min_distance < 1e-9:
        raise ValueError(
            f"Points are not distinct: minimum distance = {min_distance:.2e} < 1e-9"
        )

    max_distance = np.max(distances)

    ratio = max_distance / min_distance
    fitness = ratio ** 2

    return {
        "fitness": float(fitness),
        "is_valid": 1,
        "max_distance": float(max_distance),
        "min_distance": float(min_distance),
    }
