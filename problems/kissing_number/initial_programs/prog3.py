import numpy as np


def entrypoint() -> np.ndarray:
    np.random.seed(42)
    points = []
    for _ in range(10):
        vec = np.zeros(12, dtype=int)
        indices = np.random.choice(12, 3, replace=False)
        for idx in indices:
            vec[idx] = np.random.choice([-1, 1])
        points.append(vec.tolist())
    return np.array(points, dtype=int)
