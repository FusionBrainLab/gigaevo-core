"""
Validation function for: Matrix multiplication tensor decomposition for n=2, m=4, p=5 to minimize rank
"""

import numpy as np

from helper import (
    BENCHMARK_RANK,
    SUCCESS_THRESHOLD,
    compute_combined_score,
    verify_tensor_decomposition,
)


def validate(decomposition, n, m, p, loss, rank):
    """
    Validate the solution and compute fitness metrics.

    Args:
        decomposition: tuple of factor matrices from entrypoint(), n,m,p: ints, loss: float, rank: int

    Returns:
        dict with metrics:
        - combined_score: Inverse rank benchmark: 32 / rank (PRIMARY OBJECTIVE - maximize, > 1 means new record)
        - loss: Final reconstruction loss (must be <= 1e-6 for success)
        - rank: Rank of the discovered tensor decomposition
        - eval_time: Execution time in seconds (not available here, set to 0.0)
        - is_valid: Whether the program is valid (1 valid, 0 invalid)
    """
    try:
        n = int(n)
        m = int(m)
        p = int(p)
        rank = int(rank)
        loss = float(loss)

        if not isinstance(decomposition, (tuple, list)) or len(decomposition) != 3:
            raise ValueError("decomposition must be a tuple with three factor matrices.")

        np_decomposition = tuple(np.asarray(factor, dtype=np.float64) for factor in decomposition)

        verify_tensor_decomposition(np_decomposition, n, m, p, rank)

        is_valid = 1 if loss <= SUCCESS_THRESHOLD else 0
        combined_score = compute_combined_score(rank) if is_valid else 0.0

        return {
            "combined_score": float(combined_score),
            "loss": float(loss),
            "rank": float(rank),
            "eval_time": 0.0,
            "is_valid": is_valid,
        }
    except Exception:
        return {
            "combined_score": 0.0,
            "loss": 1.0,
            "rank": 0.0,
            "eval_time": 0.0,
            "is_valid": 0,
        }