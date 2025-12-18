import numpy as np


def entrypoint() -> np.ndarray:
    np.random.seed(42)
    points = np.random.rand(14, 3)
    return points

