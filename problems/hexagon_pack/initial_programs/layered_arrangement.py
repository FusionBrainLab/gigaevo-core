"""
Layered Arrangement Strategy for Hexagon Packing

This strategy creates a central core with surrounding layers,
providing structural stability and efficient space utilization.
"""

import numpy as np
import random
from helper import compute_outer_hex_side_length, get_hexagon_vertices, check_hexagon_overlap_many, check_hexagon_overlap_two

def entrypoint() -> tuple[np.ndarray, np.ndarray]:
    min_distance = 2.4  
    
    core = np.array([[0.0, 0.0]])
    
    layer1_radius = 2.6  
    layer1 = []
    for i in range(6):
        angle = i * np.pi / 3
        layer1.append([layer1_radius * np.cos(angle), layer1_radius * np.sin(angle)])
    
    layer2_radius = 5.0  
    layer2 = []
    for i in range(4):
        angle = i * np.pi / 2 + np.pi / 4
        layer2.append([layer2_radius * np.cos(angle), layer2_radius * np.sin(angle)])
    
    centers = np.vstack([core, np.array(layer1), np.array(layer2)])
    
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < min_distance:
                print(f"Warning: centers {i} and {j} too close: {dist}")
    
    angles = np.array([0.0,
                      np.pi/6, np.pi/3, 0.0, -np.pi/6, -np.pi/3, np.pi/2,
                      np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4])
    
    return centers, angles 