import numpy as np

from helper import build_matmul_tensor, canonical_decomposition


def _try_reduce_rank(decomposition: tuple[np.ndarray, np.ndarray, np.ndarray], target_rank: int):
    """Attempt to keep only the first target_rank columns if they still reconstruct exactly."""
    U, V, W = decomposition
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return decomposition

    reduced = (U[:, :target_rank], V[:, :target_rank], W[:, :target_rank])
    return reduced


def entrypoint():
    """
    Grid/heuristic search over low-rank factor candidates

    Returns:
        (decomposition, n, m, p, loss, rank) where decomposition = (U, V, W) factor matrices
    """
    n, m, p = 2, 4, 5
    target_tensor = build_matmul_tensor(n, m, p)

    # Begin with the canonical exact solution (rank 40).
    best_decomposition = canonical_decomposition(n, m, p)
    best_rank = best_decomposition[0].shape[1]

    # Try a few deterministic smaller ranks by truncating columns and checking reconstruction.
    for candidate_rank in [32, 36, 38]:
        reduced = _try_reduce_rank(best_decomposition, candidate_rank)
        reconstructed = np.einsum("ir,jr,kr->ijk", *reduced)
        if np.array_equal(reconstructed, target_tensor):
            best_decomposition = reduced
            best_rank = candidate_rank

    reconstructed = np.einsum("ir,jr,kr->ijk", *best_decomposition)
    loss = float(np.max(np.abs(reconstructed - target_tensor)))

    return best_decomposition, n, m, p, loss, best_rank