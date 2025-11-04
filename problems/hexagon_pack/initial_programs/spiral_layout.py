"""
Spiral Layout Strategy for Hexagon Packing

This strategy arranges hexagons in a golden spiral pattern for
optimal space utilization based on natural growth patterns.
"""

import numpy as np


def entrypoint() -> tuple[np.ndarray, np.ndarray]:
    golden_angle = np.pi * (3 - np.sqrt(5))
    min_distance = 2.4

    centers = []
    angles = []

    for i in range(11):
        r = 2.0 * np.sqrt(i + 1)
        theta = i * golden_angle
        candidate_center = np.array([r * np.cos(theta), r * np.sin(theta)])

        for existing_center in centers:
            if np.linalg.norm(candidate_center - existing_center) < min_distance:
                r *= 1.3
                candidate_center = np.array([r * np.cos(theta), r * np.sin(theta)])
                break

        centers.append(candidate_center)
        angles.append(theta + np.pi / 6)

    return np.array(centers), np.array(angles)
