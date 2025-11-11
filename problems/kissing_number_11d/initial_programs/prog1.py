import numpy as np


def entrypoint() -> np.ndarray:
    np.random.seed(42)
    points = np.array(
        [
            [1] * 11,
            [-1] * 11,
        ],
        dtype=int,
    )
    return points
