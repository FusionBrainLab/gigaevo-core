import numpy as np

def entrypoint() -> np.ndarray:
    np.random.seed(42)
    points = []
    for i in range(12):
        vec = [0] * 12
        vec[i] = 2
        points.append(vec)
        neg_vec = [0] * 12
        neg_vec[i] = -2
        points.append(neg_vec)
    return np.array(points, dtype=int)