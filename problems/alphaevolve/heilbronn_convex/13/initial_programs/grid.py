from helper import *
import numpy as np


def entrypoint():
    """
    Regular grid-based point arrangement

    Returns:
        (13, 2) NumPy array of (x, y) coordinates for 13 points
    """
    n = 13
    
    # Use a 4x4 grid (16 points) and select 13 points
    # Arrange in a roughly square grid
    grid_size = 4
    x_coords = np.linspace(0.1, 0.9, grid_size)
    y_coords = np.linspace(0.1, 0.9, grid_size)
    
    # Create grid points
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            grid_points.append([x, y])
    
    # Select 13 points from the grid (prefer interior points)
    # Use a deterministic selection
    selected_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15]  # Skip some edge points
    points = np.array([grid_points[i] for i in selected_indices[:n]])
    
    return points
