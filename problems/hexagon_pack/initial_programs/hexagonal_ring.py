"""
Hexagonal Ring Strategy for Hexagon Packing

This strategy arranges hexagons in an inner cluster with an outer ring
formation, mimicking natural hexagonal packing patterns.
"""

import numpy as np


def entrypoint() -> tuple[np.ndarray, np.ndarray]:
    min_distance = 2.4

    inner_centers = np.array([[0.0, 0.0], [2.4, 0.0], [1.2, 2.08]])

    ring_radius = 4.5
    outer_centers = []
    for i in range(8):
        angle = i * 2 * np.pi / 8
        outer_centers.append(
            [ring_radius * np.cos(angle), ring_radius * np.sin(angle)]
        )

    centers = np.vstack([inner_centers, np.array(outer_centers)])

    for i in range(3, len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < min_distance:
                ring_radius += 0.5
                break

    angles = np.array(
        [
            0.0,
            np.pi / 3,
            np.pi / 6,
            0.0,
            np.pi / 4,
            np.pi / 2,
            -np.pi / 4,
            -np.pi / 2,
            np.pi / 8,
            -np.pi / 8,
            np.pi / 12,
        ]
    )

    return centers, angles
