import numpy as np


def entrypoint() -> np.ndarray:
    np.random.seed(42)
    n = 26
    centers = []
    radii = []
    grid_size_x = 5
    grid_size_y = 6
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            if len(centers) >= n:
                break
            x = (i + 1) / (grid_size_x + 1)
            y = (j + 1) / (grid_size_y + 1)
            max_r = min(1 / (grid_size_x + 1), 1 / (grid_size_y + 1)) * 0.4
            r = max_r * (0.7 + 0.3 * np.random.rand())
            centers.append([x, y])
            radii.append(r)
        if len(centers) >= n:
            break
    
    centers = np.array(centers, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32)
    result = np.column_stack([centers, radii])
    return result

