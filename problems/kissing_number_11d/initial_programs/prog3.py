import numpy as np


def entrypoint() -> np.ndarray:
    np.random.seed(42)
    points = []
    for _ in range(100):
        vec = np.zeros(11, dtype=int)
        indices = np.random.choice(11, 3, replace=False)
        for idx in indices:
            vec[idx] = np.random.choice([-1, 1])
        points.append(vec.tolist())
    return np.array(points, dtype=int)
