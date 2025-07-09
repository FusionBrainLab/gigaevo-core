"""
Tight Cluster Strategy for Hexagon Packing

This strategy places hexagons close together in multiple clusters 
with proper spacing validation to avoid overlaps.
"""

import numpy as np
import random
from helper import compute_outer_hex_side_length, get_hexagon_vertices, check_hexagon_overlap_many, check_hexagon_overlap_two


def entrypoint() -> tuple[np.ndarray, np.ndarray]:
    min_distance = 2.4  

    centers = np.array([
        [0.0, 0.0],
        [2.5, 0.0],
        [-1.3, 2.2],
        [-1.3, -2.2],
        [1.3, 2.2],
        [5.0, 0.0],
        [2.5, 3.5],
        [2.5, -3.5],
        [-3.5, 0.0],
        [-2.5, 4.0],
        [-2.5, -4.0]
    ])
    
    max_iterations = 10
    for iteration in range(max_iterations):
        needs_adjustment = False
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_distance:
                    direction = centers[j] - centers[i]
                    if np.linalg.norm(direction) > 1e-8:
                        centers[j] = centers[i] + direction / np.linalg.norm(direction) * min_distance * 1.1
                        needs_adjustment = True
        
        if not needs_adjustment:
            break
    
    angles = np.array([0.0, np.pi/6, np.pi/3, -np.pi/6, np.pi/2, 
                      np.pi/4, -np.pi/4, np.pi/8, -np.pi/8, np.pi/12, -np.pi/12])
    
    return centers, angles 