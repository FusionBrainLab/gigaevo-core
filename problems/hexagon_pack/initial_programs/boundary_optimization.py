"""
Boundary Optimization Strategy for Hexagon Packing

This strategy focuses on boundary-optimized arrangements with
central stability to minimize the enclosing hexagon size.
"""

import numpy as np


def entrypoint() -> tuple[np.ndarray, np.ndarray]:
    min_distance = 2.4

    core = np.array([[0.0, 0.0], [2.4, 0.0], [1.2, 2.08]])

    boundary_positions = [
        [4.8, 1.0],
        [-2.4, 1.0],
        [1.2, 4.5],
        [1.2, -2.5],
        [4.0, -2.0],
        [-1.2, -2.5],
        [-3.6, 3.0],
        [3.6, 3.5],
    ]

    centers = np.vstack([core, np.array(boundary_positions)])

    # Validate all distances and adjust if necessary
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < min_distance:
                # Push apart if too close
                direction = centers[j] - centers[i]
                centers[j] = (
                    centers[i]
                    + direction / np.linalg.norm(direction) * min_distance * 1.1
                )

    # Rotation strategy: optimize for boundary contact
    angles = np.array(
        [
            0.0,
            np.pi / 3,
            2 * np.pi / 3,  # Core rotations
            np.pi / 8,
            -np.pi / 8,
            np.pi / 4,
            -np.pi / 4,
            3 * np.pi / 8,
            -3 * np.pi / 8,
            np.pi / 2,
            -np.pi / 2,
        ]
    )  # Boundary

    return centers, angles
