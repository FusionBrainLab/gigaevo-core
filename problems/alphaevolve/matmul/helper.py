"""
Helper functions for: Matrix multiplication tensor decomposition for n=2, m=4, p=5 to minimize rank

Utility functions shared between validate.py and user programs.
These functions are importable via: from helper import function_name
"""

import numpy as np

BENCHMARK_RANK = 32
SUCCESS_THRESHOLD = 1e-6


def build_matmul_tensor(n: int, m: int, p: int, dtype=np.float64) -> np.ndarray:
    """Construct the canonical matrix multiplication tensor for <n, m, p>."""
    tensor = np.zeros((n * m, m * p, n * p), dtype=dtype)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i * m + j, j * p + k, k * n + i] = 1
    return tensor


def verify_tensor_decomposition(
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray],
    n: int,
    m: int,
    p: int,
    rank: int,
) -> None:
    """Verify the decomposition (U, V, W) reconstructs the matmul tensor exactly."""
    if not isinstance(decomposition, (tuple, list)) or len(decomposition) != 3:
        raise ValueError("Decomposition must be a tuple of three NumPy arrays.")

    U, V, W = (np.asarray(arr) for arr in decomposition)

    if U.shape != (n * m, rank):
        raise ValueError(f"U must have shape {(n * m, rank)}, got {U.shape}")
    if V.shape != (m * p, rank):
        raise ValueError(f"V must have shape {(m * p, rank)}, got {V.shape}")
    if W.shape != (n * p, rank):
        raise ValueError(f"W must have shape {(n * p, rank)}, got {W.shape}")

    matmul_tensor = build_matmul_tensor(n, m, p, dtype=np.float64)
    reconstructed = np.einsum("ir,jr,kr->ijk", U, V, W)

    if not np.array_equal(reconstructed, matmul_tensor):
        diff = np.max(np.abs(reconstructed - matmul_tensor))
        raise ValueError(
            "Tensor constructed by decomposition does not match the target tensor. "
            f"Maximum absolute difference: {diff:.6e}"
        )


def compute_combined_score(rank: int) -> float:
    """Return the benchmarked inverse rank score."""
    if rank <= 0:
        return 0.0
    return BENCHMARK_RANK / float(rank)


def canonical_decomposition(n: int, m: int, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the canonical (n*m*p)-rank decomposition that always works."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((n * p, rank), dtype=np.float64)

    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1

    return U, V, W