import numpy as np


def validate(output):
    if not isinstance(output, np.ndarray):
        raise ValueError("Output must be a NumPy ndarray.")
    if output.ndim != 2:
        raise ValueError(f"Output must be 2D (got ndim={output.ndim}).")
    n, d = output.shape
    if d != 11:
        raise ValueError(
            f"Each vector must have 11 dimensions (got shape={output.shape})."
        )
    if not np.issubdtype(output.dtype, np.integer):
        raise ValueError(f"Array dtype must be integer (got {output.dtype}).")

    # Distinctness and non-zero
    tuples = [tuple(row.tolist()) for row in output]
    if len(set(tuples)) != len(tuples):
        raise ValueError("All vectors must be distinct (found duplicates).")
    for i, row in enumerate(output):
        if not np.any(row):
            raise ValueError(
                f"Zero vector found at index {i}. All vectors must be non-zero."
            )

    # Compute squared norms and pairwise squared distances using Python ints
    def sq_norm(vec):
        return sum(int(x) * int(x) for x in vec.tolist())

    def sq_dist(a, b):
        return sum(
            int(ai - bi) * int(ai - bi) for ai, bi in zip(a.tolist(), b.tolist())
        )

    if n == 0:
        raise ValueError("Empty set provided. Need at least one non-zero vector.")

    r2_list = [sq_norm(output[i]) for i in range(n)]

    if len(set(r2_list)) != 1:
        raise ValueError(
            f"Single-shell violation: norms not all equal (found values {set(r2_list)})."
        )

    r2 = r2_list[0]

    d2_min = None
    for i in range(n):
        for j in range(i + 1, n):
            d2 = sq_dist(output[i], output[j])
            if d2_min is None or d2 < d2_min:
                d2_min = d2

    if d2_min is None:
        raise ValueError(
            "A single vector (N=1) is insufficient. Need at least two vectors to define pairwise separation."
        )

    if not (d2_min >= r2):
        raise ValueError(
            "Separation certificate failed: require d2_min ≥ r2.\n"
            f"Details: r2={r2}, d2_min={d2_min}, margin={d2_min - r2}."
        )

    return {
        "fitness": int(n),
        "margin": int(d2_min - r2),
        "is_valid": 1,
    }
