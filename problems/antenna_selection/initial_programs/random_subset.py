r"""Diversity baseline: keep a random subset of K antennas.

The RNG is seeded deterministically from the matrix contents so the same matrix
always yields the same mask (keeps fitness reproducible across re-evaluation).
"""

import numpy as np


class Solver:
    def solve(self, V, K, sigma):
        n = V.shape[0]
        seed = int(np.abs(V).sum() * 1e6) % (2**32)
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=K, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[keep] = True
        return mask


def entrypoint():
    return Solver()
