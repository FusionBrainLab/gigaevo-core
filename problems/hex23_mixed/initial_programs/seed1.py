r"""Trivial identity seed: propose the aligned honeycomb tiling from scratch,
return any supplied incumbent unchanged.

Cells $c = i\,a_1 + j\,a_2$ with $a_1 = \sqrt{3}(\cos 30^\circ, \sin 30^\circ)$,
$a_2 = (0, \sqrt{3})$ and rotation $0$ tile the plane exactly, so any subset is
non-overlapping. A flat-topped container of side $L$ contains an aligned unit
hexagon at $c$ iff $c \cdot n_k \le (L-1)\sqrt{3}/2$ for its six edge normals
$n_k$, so keeping the `hex_num` cells with smallest $\max_k c \cdot n_k$ gives
$L = \frac{2}{\sqrt{3}} \max_i \max_k c_i \cdot n_k + 1$, minimal over subsets
of this tiling. Deterministic: every call returns the same configuration.
"""

import numpy as np

_NORMALS = np.stack(
    [np.array([np.cos(a), np.sin(a)]) for a in np.pi / 6 + np.pi / 3 * np.arange(6)]
)


def _lattice_config(hex_num: int) -> tuple[np.ndarray, np.ndarray]:
    a1 = np.sqrt(3.0) * np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    a2 = np.array([0.0, np.sqrt(3.0)])
    reach = 6
    cells = []
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            c = i * a1 + j * a2
            cells.append((float(np.max(_NORMALS @ c)), i, j))
    cells.sort()
    centers = np.array(
        [i * a1 + j * a2 for _, i, j in cells[:hex_num]], dtype=np.float64
    )
    return centers, np.zeros(hex_num, dtype=np.float64)


class Proposer:
    def __init__(self, hex_num: int = 23, seed: int = 0):
        self.hex_num = hex_num
        self.seed = seed

    def propose(
        self,
        input_config: tuple[np.ndarray, np.ndarray] | None,
        intensity: float,
        seed=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if input_config is None:
            return _lattice_config(self.hex_num)
        centers, angles = input_config
        return np.asarray(centers, dtype=np.float64), np.asarray(
            angles, dtype=np.float64
        )


def entrypoint():
    return Proposer
