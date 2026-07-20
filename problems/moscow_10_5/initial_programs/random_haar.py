"""Generic dense subspace: deterministic Haar-distributed baseline."""

import numpy as np


def entrypoint() -> np.ndarray:
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((10, 5))
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return basis
