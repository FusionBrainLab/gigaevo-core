from helper import *
import numpy as np


def entrypoint():
    """
    Uniform step function with proper normalization

    Returns:
        (h_values, c5_bound, n_points) where h_values is (n_points,) NumPy array, c5_bound is float, n_points is int
    """
    n_points = 200
    dx = 2.0 / n_points
    
    # Create uniform step function: h(x) = 0.5 for all x
    # To satisfy integral = 1, we need: sum(h) * dx = 1
    # So: sum(h) = 1 / dx = n_points / 2
    # For uniform h, each value should be: (n_points / 2) / n_points = 0.5
    h_values = np.full(n_points, 0.5)
    
    # Compute C5 bound
    c5_bound = compute_c5_bound(h_values, n_points)
    
    return h_values, c5_bound, n_points
