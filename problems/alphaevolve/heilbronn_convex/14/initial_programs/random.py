from helper import *
import numpy as np


def entrypoint():
    """
    Random point placement with fixed seed

    Returns:
        (14, 2) NumPy array of (x, y) coordinates for 14 points
    """
    n = 14
    rng = np.random.default_rng(seed=42)
    points = rng.random((n, 2))
    return points
