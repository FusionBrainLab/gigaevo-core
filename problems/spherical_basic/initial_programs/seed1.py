"""Trivial identity seed: return the warm start unchanged, so mu == mu_cohn and fitness == 0."""

import numpy as np


class Solver:
    def __init__(self, n: int, d: int, seed: int = 0):
        self.n = n
        self.d = d

    def solve(self, points: np.ndarray, seed=None) -> np.ndarray:
        return np.asarray(points, dtype=np.float64)


def entrypoint():
    return Solver
