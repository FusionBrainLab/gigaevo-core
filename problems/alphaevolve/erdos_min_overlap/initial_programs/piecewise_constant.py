from helper import *
import numpy as np


def entrypoint():
    """
    Piecewise constant function with optimization

    Returns:
        (h_values, c5_bound, n_points) where h_values is (n_points,) NumPy array, c5_bound is float, n_points is int
    """
    n_points = 200
    dx = 2.0 / n_points
    
    # Create a piecewise constant function
    # Split domain into two regions: [0, 1] and [1, 2]
    # Use different constant values in each region
    mid_point = n_points // 2
    
    # First half: higher value
    # Second half: lower value
    # Normalize to satisfy integral = 1
    # Let h1 be value in [0, 1] and h2 be value in [1, 2]
    # Then: h1 * 1 + h2 * 1 = 1 (since each region has width 1)
    # Choose h1 = 0.6, h2 = 0.4
    h1 = 0.6
    h2 = 0.4
    
    h_values = np.zeros(n_points)
    h_values[:mid_point] = h1
    h_values[mid_point:] = h2
    
    # Normalize to ensure integral = 1
    current_integral = np.sum(h_values) * dx
    h_values = h_values / current_integral
    
    # Compute C5 bound
    c5_bound = compute_c5_bound(h_values, n_points)
    
    return h_values, c5_bound, n_points
