"""
Boundary Optimization Strategy for Hexagon Packing

This strategy focuses on boundary-optimized arrangements with 
central stability to minimize the enclosing hexagon size.
"""

import numpy as np

def construct_packing() -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy adaptive placement of 9 non-overlapping circles inside the unit square.
    Random centers are proposed and radii are shrunk until valid.

    Returns:
        centers: (9, 2) ndarray of circle centers
        radii:   (9,) ndarray of circle radii
    """
    n = 9
    centers = []
    radii = []

    def is_valid(new_center, new_radius, existing_centers, existing_radii):
        # Inside square
        if not (new_radius <= new_center[0] <= 1 - new_radius and new_radius <= new_center[1] <= 1 - new_radius):
            return False
        # No overlap
        for c, r in zip(existing_centers, existing_radii):
            dist = np.linalg.norm(np.array(c) - new_center)
            if dist < (r + new_radius):
                return False
        return True

    max_attempts = 1000
    while len(centers) < n and max_attempts > 0:
        candidate_center = np.random.rand(2)
        candidate_radius = 0.2

        # Shrink radius until it fits
        while candidate_radius > 0.01:
            if is_valid(candidate_center, candidate_radius, centers, radii):
                centers.append(candidate_center.tolist())
                radii.append(candidate_radius)
                break
            candidate_radius *= 0.95

        max_attempts -= 1

    return np.array(centers), np.array(radii)