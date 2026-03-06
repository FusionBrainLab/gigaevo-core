import numpy as np


def entrypoint() -> np.ndarray:
    np.random.seed(42)
    points = np.random.rand(16, 2)
    return points
