"""
Validation function for: First Autocorrelation Inequality Problem - finding optimal non-negative function to minimize upper bound for constant C1
"""

from helper import *
import numpy as np

BENCHMARK = 1.5052939684401607
TOL = 1e-6


def validate(f_values, c1_achieved, loss, n_points):
    """
    Validate the solution and compute fitness metrics.

    Args:
        f_values: (n_points,) array from entrypoint()
        c1_achieved: float from entrypoint()
        loss: float from entrypoint()
        n_points: int from entrypoint()

    Returns:
        dict with metrics:
        - c1: The upper bound for constant C1 found by the program (PRIMARY OBJECTIVE - minimize)
        - combined_score: Progress toward benchmark: 1.5052939684401607 / c1 (PRIMARY OBJECTIVE - maximize, > 1 means new record)
        - loss: Loss value used in minimization
        - n_points: Number of points used in the discretization
        - is_valid: Whether the program is valid (1 valid, 0 invalid)
    """
    try:
        # Convert to numpy array if needed
        if not isinstance(f_values, np.ndarray):
            f_values = np.array(f_values)

        # Ensure n_points is an integer
        n_points = int(n_points)

        # Verify the solution constraints
        verify_autocorrelation_solution(f_values, float(c1_achieved), n_points)

        # Compute combined_score
        combined_score = BENCHMARK / float(c1_achieved)

        return {
            "c1": float(c1_achieved),
            "combined_score": float(combined_score),
            "loss": float(loss),
            "n_points": int(n_points),
            "is_valid": 1,
        }
    except Exception as e:
        return {
            "c1": 10.0,
            "combined_score": 0.0,
            "loss": 10.0,
            "n_points": 0,
            "is_valid": 0,
        }

