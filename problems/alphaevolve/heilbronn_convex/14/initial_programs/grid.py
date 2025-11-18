from helper import *
import numpy as np


def entrypoint():
    """
    Regular grid-based point arrangement

    Returns:
        (14, 2) NumPy array of (x, y) coordinates for 14 points
    """
    n = 14
    
    # Use a 4x4 grid (16 points) and select 14 points
    # Arrange in a roughly square grid
    grid_size = 4
    x_coords = np.linspace(0.1, 0.9, grid_size)
    y_coords = np.linspace(0.1, 0.9, grid_size)
    
    # Create grid points
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            grid_points.append([x, y])
    
    # Select 14 points from the grid (prefer interior points)
    # Use a deterministic selection - take all but 2 edge points
    selected_indices = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15]  # Skip 2 edge points
    points = np.array([grid_points[i] for i in selected_indices[:n]])
    
    return points
