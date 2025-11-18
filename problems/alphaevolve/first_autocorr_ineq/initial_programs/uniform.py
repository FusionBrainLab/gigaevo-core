from helper import *
import numpy as np


def entrypoint():
    """
    Uniform function with proper normalization

    Returns:
        (f_values, c1_achieved, loss, n_points) where f_values is (n_points,) NumPy array, c1_achieved is float, loss is float, n_points is int
    """
    n_points = 600
    dx = 0.5 / n_points
    
    # Create uniform function: f(x) = constant for all x
    # Use a simple constant value
    constant_value = 1.0
    f_values = np.full(n_points, constant_value)
    
    # Ensure non-negativity
    f_values = np.maximum(f_values, 0.0)
    
    # Compute C1 bound
    c1_bound = compute_c1_bound(f_values, n_points)
    loss = c1_bound
    
    return f_values, c1_bound, loss, n_points

