import numpy as np


def entrypoint():
    rng = np.random.default_rng(seed=42)
    points = rng.random((14, 2)).astype(np.float32)
    return points

