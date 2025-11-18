"""
Helper functions for: Heilbronn Triangle Problem for Convex Regions - optimal placement of 13 points to maximize minimum triangle area

Utility functions shared between validate.py and user programs.
These functions are importable via: from helper import function_name
"""

import numpy as np
import itertools
from scipy.spatial import ConvexHull


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the area of a triangle given its vertices p1, p2, and p3.

    Args:
        p1: First vertex as (x, y) array
        p2: Second vertex as (x, y) array
        p3: Third vertex as (x, y) array

    Returns:
        Area of the triangle
    """
    return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2


def compute_min_area_normalized(points: np.ndarray) -> float:
    """Compute the minimum triangle area normalized by convex hull area.

    Args:
        points: (n, 2) array of point coordinates

    Returns:
        Minimum triangle area divided by convex hull area
    """
    # Compute minimum triangle area
    min_triangle_area = min(
        [triangle_area(p1, p2, p3) for p1, p2, p3 in itertools.combinations(points, 3)]
    )

    # Compute convex hull area
    convex_hull_area = ConvexHull(points).volume

    # Normalize
    if convex_hull_area < 1e-10:
        return 0.0

    return float(min_triangle_area / convex_hull_area)


def validate_points(points: np.ndarray, num_points: int = 13) -> None:
    """Validate that points meet the geometric constraints.

    Args:
        points: (n, 2) array of point coordinates
        num_points: Expected number of points

    Raises:
        ValueError: If any constraint is violated
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    if points.shape != (num_points, 2):
        raise ValueError(f"Invalid shapes: points = {points.shape}, expected {(num_points, 2)}")

    # Check for finite coordinates
    if not np.all(np.isfinite(points)):
        raise ValueError("All coordinates must be finite real numbers")

    # Check for duplicate points (within numerical tolerance)
    tol = 1e-10
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < tol:
                raise ValueError(f"Duplicate points found: points {i} and {j} are too close (distance: {dist})")

    # Check that all points are within or on the convex hull
    # (This is automatically satisfied since we compute the convex hull of the points themselves)
