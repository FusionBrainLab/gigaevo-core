"""Trivial identity seed: return the incumbent unchanged, so mu == mu_cohn and fitness == 0."""

import numpy as np


class Proposer:
    def __init__(self, n: int, d: int, seed: int = 0):
        self.n = n
        self.d = d

    def propose(self, input_config, intensity: float, seed=None) -> np.ndarray:
        if input_config is None:
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((self.n, self.d))
            return x / np.linalg.norm(x, axis=1, keepdims=True)
        return np.asarray(input_config, dtype=np.float64)


def entrypoint():
    return Proposer
