"""
Helper functions for: First Autocorrelation Inequality Problem - finding optimal non-negative function to minimize upper bound for constant C1

Utility functions shared between validate.py and user programs.
These functions are importable via: from helper import function_name
"""

import numpy as np


def verify_autocorrelation_solution(f_values: np.ndarray, c1_achieved: float, n_points: int):
    """Verify the autocorrelation solution for UPPER BOUND optimization.

    Args:
        f_values: (n_points,) array of function values
        c1_achieved: The reported C1 bound
        n_points: Number of discretization points

    Raises:
        ValueError: If any constraint is violated or values don't match
    """
    # Check shape
    if f_values.shape != (n_points,):
        raise ValueError(f"Expected function values shape {(n_points,)}. Got {f_values.shape}.")

    # Check non-negativity
    if np.any(f_values < 0.0):
        raise ValueError("Function must be non-negative.")

    # Recompute C1 to verify
    dx = 0.5 / n_points
    f_nonneg = np.maximum(f_values, 0.0)

    # Compute the FULL autoconvolution
    autoconv = np.convolve(f_nonneg, f_nonneg, mode="full") * dx

    # The rest of the calculation can be simplified as we now take the max over the whole result
    integral_sq = (np.sum(f_nonneg) * dx) ** 2

    if integral_sq < 1e-8:
        raise ValueError("Function integral is too small.")

    # The max of the full autoconv is the correct value
    computed_c1 = float(np.max(autoconv / integral_sq))

    # Verify consistency
    delta = abs(computed_c1 - c1_achieved)
    if delta > 1e-6:
        raise ValueError(
            f"C1 mismatch: reported {c1_achieved:.6f}, computed {computed_c1:.6f}, delta: {delta:.6f}"
        )


def compute_c1_bound(f_values: np.ndarray, n_points: int) -> float:
    """Compute the C1 bound from function values.

    Args:
        f_values: (n_points,) array of function values
        n_points: Number of discretization points

    Returns:
        The computed C1 bound
    """
    dx = 0.5 / n_points
    f_nonneg = np.maximum(f_values, 0.0)

    # Compute the FULL autoconvolution
    autoconv = np.convolve(f_nonneg, f_nonneg, mode="full") * dx

    # Compute integral squared
    integral_sq = (np.sum(f_nonneg) * dx) ** 2

    if integral_sq < 1e-8:
        return float('inf')  # Return infinity if integral is too small

    # The max of the full autoconv divided by integral squared
    return float(np.max(autoconv / integral_sq))

