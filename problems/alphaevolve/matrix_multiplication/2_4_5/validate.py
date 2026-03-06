import numpy as np


def validate(result):
    if not isinstance(result, dict):
        raise ValueError("Result must be a dictionary")
    
    required_keys = ['rank', 'u_vectors', 'v_vectors', 'w_vectors']
    for key in required_keys:
        if key not in result:
            raise ValueError(f"Missing required key: {key}")
    
    rank = result['rank']
    u_vectors = np.asarray(result['u_vectors'], dtype=float)
    v_vectors = np.asarray(result['v_vectors'], dtype=float)
    w_vectors = np.asarray(result['w_vectors'], dtype=float)
    
    n, m, p = 2, 4, 5
    
    if not isinstance(rank, (int, np.integer)) or rank <= 0:
        raise ValueError(f"Rank must be a positive integer, got {rank}")
    
    if u_vectors.ndim != 2 or u_vectors.shape[1] != n * m:
        raise ValueError(
            f"u_vectors must have shape (rank, {n*m}), got {u_vectors.shape}"
        )
    
    if v_vectors.ndim != 2 or v_vectors.shape[1] != m * p:
        raise ValueError(
            f"v_vectors must have shape (rank, {m*p}), got {v_vectors.shape}"
        )
    
    if w_vectors.ndim != 2 or w_vectors.shape[1] != n * p:
        raise ValueError(
            f"w_vectors must have shape (rank, {n*p}), got {w_vectors.shape}"
        )
    
    if u_vectors.shape[0] != rank or v_vectors.shape[0] != rank or w_vectors.shape[0] != rank:
        raise ValueError(
            f"All vectors must have first dimension equal to rank {rank}"
        )
    
    if not np.all(np.isfinite(u_vectors)) or not np.all(np.isfinite(v_vectors)) or not np.all(np.isfinite(w_vectors)):
        raise ValueError("All vectors must contain finite values")
    
    matmul_tensor = np.zeros((n * m, m * p, n * p), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                matmul_tensor[i * m + j, j * p + k, k * n + i] = 1
    
    u_reshaped = u_vectors.T
    v_reshaped = v_vectors.T
    w_reshaped = w_vectors.T
    
    constructed_tensor = np.einsum("ir,jr,kr -> ijk", u_reshaped, v_reshaped, w_reshaped)
    
    if not np.array_equal(constructed_tensor, matmul_tensor):
        diff = np.max(np.abs(constructed_tensor - matmul_tensor))
        raise ValueError(
            f"Tensor constructed by decomposition does not exactly match the target tensor. Maximum difference is {diff:.6e}."
        )
    
    BENCHMARK = 32
    fitness = BENCHMARK / float(rank)
    is_valid = 1
    
    return {
        "fitness": fitness,
        "is_valid": is_valid,
    }

