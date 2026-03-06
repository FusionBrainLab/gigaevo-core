import numpy as np


def entrypoint() -> np.ndarray:
    np.random.seed(42)
    n = 32
    centers = []
    radii = []
    for _ in range(n):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        max_r = min(x, 1 - x, y, 1 - y)
        r = np.random.uniform(0.01, max_r * 0.5)
        centers.append([x, y])
        radii.append(r)
    
    centers = np.array(centers, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32)
    result = np.column_stack([centers, radii])
    return result

