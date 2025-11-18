"""
Validation function for: Erdős Minimum Overlap Problem - finding optimal step function to minimize upper bound for constant C₅
"""

from helper import *
import numpy as np

BENCHMARK = 0.38092303510845016
TOL = 1e-6


def validate(h_values, c5_bound, n_points):
    """
    Validate the solution and compute fitness metrics.

    Args:
        h_values: (n_points,) array from entrypoint()
        c5_bound: float from entrypoint()
        n_points: int from entrypoint()

    Returns:
        dict with metrics:
        - c5_bound: The upper bound for constant C₅ found by the program (PRIMARY OBJECTIVE - minimize)
        - combined_score: Progress toward benchmark: 0.38092303510845016 / c5_bound (PRIMARY OBJECTIVE - maximize, > 1 means new record)
        - n_points: Number of points used in the discretization
        - is_valid: Whether the program is valid (1 valid, 0 invalid)
    """
    try:
        # Convert to numpy array if needed
        if not isinstance(h_values, np.ndarray):
            h_values = np.array(h_values)

        # Ensure n_points is an integer
        n_points = int(n_points)

        # Verify the solution constraints
        verify_c5_solution(h_values, float(c5_bound), n_points)

        # Compute combined_score
        combined_score = BENCHMARK / float(c5_bound)

        return {
            "c5_bound": float(c5_bound),
            "combined_score": float(combined_score),
            "n_points": int(n_points),
            "is_valid": 1,
        }
    except Exception as e:
        return {
            "c5_bound": 1.0,
            "combined_score": 0.0,
            "n_points": 0,
            "is_valid": 0,
        }
