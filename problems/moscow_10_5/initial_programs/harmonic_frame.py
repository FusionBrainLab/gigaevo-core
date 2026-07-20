"""Balanced cyclic Fourier-frame baseline."""

import numpy as np


def entrypoint() -> np.ndarray:
    angles = 2.0 * np.pi * np.arange(10, dtype=np.float64) / 10.0
    return np.column_stack(
        [
            np.ones(10, dtype=np.float64),
            np.cos(angles),
            np.sin(angles),
            np.cos(4.0 * angles),
            np.sin(4.0 * angles),
        ]
    )
