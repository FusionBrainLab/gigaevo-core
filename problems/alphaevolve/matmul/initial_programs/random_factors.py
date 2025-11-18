import numpy as np

from helper import build_matmul_tensor, canonical_decomposition


def entrypoint():
    """
    Random factor initialization with quick loss check

    Returns:
        (decomposition, n, m, p, loss, rank) where decomposition = (U, V, W) factor matrices
    """
    n, m, p = 2, 4, 5
    target_tensor = build_matmul_tensor(n, m, p)

    # Start from the canonical exact decomposition (rank = n * m * p = 40).
    decomposition = canonical_decomposition(n, m, p)
    rank = n * m * p

    reconstructed = np.einsum("ir,jr,kr->ijk", *decomposition)
    loss = float(np.max(np.abs(reconstructed - target_tensor)))

    return decomposition, n, m, p, loss, rank