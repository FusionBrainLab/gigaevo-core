import numpy as np


def entrypoint() -> np.ndarray:
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(2):
                x = i / 2.0
                y = j / 2.0
                z = k / 1.0
                points.append([x, y, z])
    points = points[:14]
    
    return np.array(points, dtype=np.float32)

