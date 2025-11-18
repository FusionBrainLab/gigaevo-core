"""
Helper functions for: Erdős Minimum Overlap Problem - finding optimal step function to minimize upper bound for constant C₅

Utility functions shared between validate.py and user programs.
These functions are importable via: from helper import function_name
"""

import numpy as np


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    """Verifies the C5 upper bound solution.

    Args:
        h_values: (n_points,) array of step function values
        c5_achieved: The reported C5 bound
        n_points: Number of discretization points

    Raises:
        ValueError: If any constraint is violated or values don't match
    """
    if h_values.shape != (n_points,):
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    # Verify h(x) in [0, 1] constraint
    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    # Verify integral of h = 1 constraint
    dx = 2.0 / n_points
    integral_h = np.sum(h_values) * dx
    if not np.isclose(integral_h, 1.0, atol=1e-3):
        raise ValueError(f"Integral of h is not close to 1. Got: {integral_h:.6f}")

    # Re-calculate the C5 bound using np.correlate
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    # Check for consistency
    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")


def compute_c5_bound(h_values: np.ndarray, n_points: int) -> float:
    """Compute the C5 bound from step function values.

    Args:
        h_values: (n_points,) array of step function values
        n_points: Number of discretization points

    Returns:
        The computed C5 bound
    """
    dx = 2.0 / n_points
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    return float(np.max(correlation))
