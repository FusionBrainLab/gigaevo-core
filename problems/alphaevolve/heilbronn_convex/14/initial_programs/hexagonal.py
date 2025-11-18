from helper import *
import numpy as np


def entrypoint():
    """
    Hexagonal close-packing arrangement

    Returns:
        (14, 2) NumPy array of (x, y) coordinates for 14 points
    """
    n = 14
    
    # Create a hexagonal arrangement
    # Use a hexagonal lattice pattern
    points = []
    
    # Center point
    center_x, center_y = 0.5, 0.5
    points.append([center_x, center_y])
    
    # First ring: 6 points around center
    radius1 = 0.25
    for i in range(6):
        angle = i * np.pi / 3
        x = center_x + radius1 * np.cos(angle)
        y = center_y + radius1 * np.sin(angle)
        points.append([x, y])
    
    # Second ring: 7 more points (to get total of 14)
    radius2 = 0.4
    for i in range(7):
        angle = i * 2 * np.pi / 7  # Distribute evenly around circle
        x = center_x + radius2 * np.cos(angle)
        y = center_y + radius2 * np.sin(angle)
        points.append([x, y])
    
    # Take first n points
    points = np.array(points[:n])
    
    # Normalize to [0, 1] range
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0) + 1e-10)
    points = points * 0.8 + 0.1  # Scale to [0.1, 0.9] range
    
    return points
