import numpy as np

def construct_packing() -> tuple[np.ndarray, np.ndarray]:
    """
    Place 9 circles on a jittered 3×3 grid inside the unit square.
    Each circle is adjusted to avoid boundary violation and potential overlap.
    
    Returns:
        centers: (9, 2) ndarray of (x, y) positions
        radii:   (9,) ndarray of radii
    """
    n = 9
    grid_size = 3
    spacing = 1.0 / (grid_size + 1)
    jitter = 0.05

    centers = []
    radii = []

    for i in range(grid_size):
        for j in range(grid_size):
            # Base grid point
            x = (i + 1) * spacing
            y = (j + 1) * spacing

            # Add small noise
            x += np.random.uniform(-jitter, jitter)
            y += np.random.uniform(-jitter, jitter)

            # Clamp inside unit square (with margin)
            r = 0.12
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)

            centers.append([x, y])
            radii.append(r)

    return np.array(centers), np.array(radii)