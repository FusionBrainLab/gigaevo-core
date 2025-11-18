import numpy as np


def entrypoint() -> np.ndarray:
    np.random.seed(42)
    points = np.array(
        [
            [1] * 12,
            [-1] * 12,
        ],
        dtype=int,
    )
    return points
