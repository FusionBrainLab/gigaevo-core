from helper import *
import numpy as np


def entrypoint():
    """
    Random point placement with fixed seed

    Returns:
        (13, 2) NumPy array of (x, y) coordinates for 13 points
    """
    n = 13
    rng = np.random.default_rng(seed=42)
    points = rng.random((n, 2))
    return points
