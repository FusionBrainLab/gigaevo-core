from helper import *
import numpy as np


def entrypoint():
    """
    Step function with optimization

    Returns:
        (f_values, c1_achieved, loss, n_points) where f_values is (n_points,) NumPy array, c1_achieved is float, loss is float, n_points is int
    """
    n_points = 600
    dx = 0.5 / n_points
    
    # Create a step function: f(x) = 1 in the middle half, 0 elsewhere
    # This is similar to the initialization in jax_optimizer
    f_values = np.zeros(n_points)
    start_idx = n_points // 4
    end_idx = 3 * n_points // 4
    f_values[start_idx:end_idx] = 1.0
    
    # Add small random perturbation
    np.random.seed(42)
    f_values += 0.05 * np.random.uniform(0, 1, n_points)
    
    # Ensure non-negativity
    f_values = np.maximum(f_values, 0.0)
    
    # Compute C1 bound
    c1_bound = compute_c1_bound(f_values, n_points)
    loss = c1_bound
    
    return f_values, c1_bound, loss, n_points

