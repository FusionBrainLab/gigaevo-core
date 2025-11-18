"""
Validation function for: Heilbronn Triangle Problem for Convex Regions - optimal placement of 13 points to maximize minimum triangle area
"""

from helper import *
import numpy as np

BENCHMARK = 0.030936889034895654
NUM_POINTS = 13
TOL = 1e-6


def validate(points):
    """
    Validate the solution and compute fitness metrics.

    Args:
        points: (13, 2) array from entrypoint()

    Returns:
        dict with metrics:
        - min_area_normalized: Area of smallest triangle divided by convex hull area (PRIMARY OBJECTIVE - maximize)
        - combined_score: Progress toward benchmark: min_area_normalized / 0.030936889034895654 (PRIMARY OBJECTIVE - maximize, > 1 means new record)
        - is_valid: Whether the program is valid (1 valid, 0 invalid)
    """
    try:
        # Convert to numpy array if needed
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        # Validate constraints
        validate_points(points, NUM_POINTS)

        # Compute metrics
        min_area_normalized = compute_min_area_normalized(points)
        combined_score = min_area_normalized / BENCHMARK

        return {
            "min_area_normalized": float(min_area_normalized),
            "combined_score": float(combined_score),
            "is_valid": 1,
        }
    except Exception as e:
        return {
            "min_area_normalized": 0.0,
            "combined_score": 0.0,
            "is_valid": 0,
        }
