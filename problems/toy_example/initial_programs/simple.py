import numpy as np
import random

def entrypoint() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a crude arrangement of 9 non-overlapping circles with variable radii,
    packed inside the unit square. Tries to prioritize center and edge usage.

    Returns:
        centers: (9, 2) ndarray of (x, y) coordinates
        radii:   (9,) ndarray of positive radii
    """
    n = 9
    centers = []
    radii = []

    # Large central circle
    centers.append([0.5, 0.5])
    radii.append(0.22)

    # Surrounding smaller circles
    angles = np.linspace(0, 2 * np.pi, n - 1, endpoint=False)
    for theta in angles:
        r = 0.11 + 0.02 * np.random.rand()  # Slight variation in radii
        x = 0.5 + 0.35 * np.cos(theta)
        y = 0.5 + 0.35 * np.sin(theta)

        # Clip to ensure within square bounds
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)

        centers.append([x, y])
        radii.append(r)

    return np.array(centers), np.array(radii)
