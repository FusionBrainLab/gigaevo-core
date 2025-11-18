import numpy as np


def validate(u_set):
    u_set = np.asarray(u_set, dtype=int)

    if u_set.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {u_set.shape}")
    if u_set.size == 0:
        raise ValueError("Array cannot be empty")
    if 0 not in u_set:
        raise ValueError("Set U must contain 0")
    if np.any(u_set < 0):
        raise ValueError("Set U must contain non-negative integers")

    u_plus_u = np.unique(u_set[:, None] + u_set[None, :])
    u_minus_u = np.unique(u_set[:, None] - u_set[None, :])

    size_U_plus_U = len(u_plus_u)
    size_U_minus_U = len(u_minus_u)
    max_U = np.max(u_set)

    if max_U == 0:
        raise ValueError("Set U must be non-trivial (max(U) > 0)")

    if size_U_minus_U > 2 * max_U + 1:
        raise ValueError(
            f"Constraint violated: |U-U| = {size_U_minus_U} > 2*max(U)+1 = {2*max_U+1}"
        )

    ratio = size_U_minus_U / size_U_plus_U
    log_ratio = np.log(ratio)
    log_denom = np.log(2 * max_U + 1)

    c6 = 1 + log_ratio / log_denom

    if not np.isfinite(c6) or c6 <= 1.0:
        raise ValueError(f"Invalid C₆ value: {c6}")

    return {"fitness": c6, "is_valid": 1}
