"""Trivial identity seed: improve/perturb return the input unchanged, so mu == mu_cohn and fitness == 0."""

import numpy as np


class Improver:
    def __init__(self, n: int, d: int, seed: int = 0):
        self.n = n
        self.d = d

    def generate_config(self, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((self.n, self.d))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def improve(self, points: np.ndarray, seed=None) -> np.ndarray:
        return np.asarray(points, dtype=np.float64)

    def perturb(self, points: np.ndarray, intensity: float, seed=None) -> np.ndarray:
        return np.asarray(points, dtype=np.float64)


def entrypoint():
    return Improver
