import numpy as np


def entrypoint() -> np.ndarray:
    points = []
    for i in range(4):
        for j in range(4):
            x = i / 3
            y = j / 3
            points.append([x, y])

    return np.array(points, dtype=np.float32)
