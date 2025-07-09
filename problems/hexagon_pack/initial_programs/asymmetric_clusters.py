"""
Asymmetric Clusters Strategy for Hexagon Packing

This strategy creates asymmetric cluster arrangements with varied spacing
to explore non-regular packing patterns and local optimizations.
"""

import numpy as np
import random
from helper import compute_outer_hex_side_length, get_hexagon_vertices, check_hexagon_overlap_many, check_hexagon_overlap_two

def entrypoint() -> tuple[np.ndarray, np.ndarray]:
    min_distance = 2.4  
    
    cluster1 = np.array([
        [0.0, 0.0],
        [2.5, 0.0],
        [1.25, 2.16],
        [-1.25, 2.16]
    ])
    
    cluster2 = np.array([
        [5.0, -1.0],
        [6.5, 1.0],
        [4.0, 2.5]
    ])
    
    cluster3 = np.array([
        [-3.5, -2.0],
        [-5.0, 0.0],
        [-2.5, -4.0],
        [-1.0, -4.5]
    ])
    
    centers = np.vstack([cluster1, cluster2, cluster3])
    
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < min_distance:
                direction = centers[j] - centers[i]
                centers[j] = centers[i] + direction / np.linalg.norm(direction) * min_distance
    
    angles = np.array([0.0, np.pi/7, 2*np.pi/7, 3*np.pi/7,
                      4*np.pi/7, 5*np.pi/7, 6*np.pi/7,
                      np.pi/5, 2*np.pi/5, 3*np.pi/5, 4*np.pi/5])
    
    return centers, angles 