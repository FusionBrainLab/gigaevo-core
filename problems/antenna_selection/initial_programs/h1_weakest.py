r"""H1 baseline: keep the K strongest antennas by power.

Turns off the antennas with the smallest per-antenna power
$s_n = \sum_j |v_{nj}|^2$, i.e. selects the $K$ rows of largest $\ell_2$ norm.
"""

import numpy as np


class Solver:
    def solve(self, V, K, sigma):
        s = np.sum(np.abs(V) ** 2, axis=1)
        keep = np.argsort(s)[::-1][:K]
        mask = np.zeros(V.shape[0], dtype=bool)
        mask[keep] = True
        return mask


def entrypoint():
    return Solver()
