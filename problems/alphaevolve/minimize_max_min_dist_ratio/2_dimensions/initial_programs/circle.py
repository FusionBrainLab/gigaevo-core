import numpy as np


def entrypoint() -> np.ndarray:
    points = []
    for i in range(16):
        angle = 2 * np.pi * i / 16
        x = 0.5 + 0.4 * np.cos(angle)
        y = 0.5 + 0.4 * np.sin(angle)
        points.append([x, y])

    return np.array(points, dtype=np.float32)
