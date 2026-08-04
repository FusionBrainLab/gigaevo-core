r"""Trivial from-scratch seed: propose the constant array $f = \mathbf{1}_n$ when
asked for a start, return any supplied incumbent unchanged.

The constant function $f = 1$ has autoconvolution $g = f * f$ equal to the discrete
triangle (ramp up to $n$, ramp back down), for which
$C = \lVert g \rVert_2^2 / (\lVert g \rVert_1 \, \lVert g \rVert_\infty) = 2/3$
exactly, independent of the resolution $n$ (both scale- and resolution-invariant).
It is the cold-start floor every arm climbs away from. Deterministic: every call
returns the same array.
"""

import numpy as np


class Proposer:
    def __init__(self, n: int = 50000, seed: int = 0):
        self.n = n
        self.seed = seed

    def propose(
        self, input_config: np.ndarray | None, intensity: float, seed=None
    ) -> np.ndarray:
        if input_config is None:
            return np.ones(self.n, dtype=np.float64)
        return np.asarray(input_config, dtype=np.float64)


def entrypoint():
    return Proposer
