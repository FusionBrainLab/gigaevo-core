import jax.numpy as jnp
import numpy as np


def get_matrix_multiplication_tensor(n, m, p):
    T = jnp.zeros((n * m, m * p, n * p), dtype=jnp.float32)
    for i, j, k in np.ndindex(n, m, p):
        T = T.at[i * m + j, j * p + k, k * n + i].set(1)
    return T


def reconstruct_tensor(u_vectors, v_vectors, w_vectors, n, m, p):
    R = u_vectors.shape[0]
    u_matrices = u_vectors.reshape(R, n, m) 
    v_matrices = v_vectors.reshape(R, m, p)  
    w_matrices = w_vectors.reshape(R, n, p) 
    
    np.random.seed(42)
    max_error = 0.0
    
    for test_idx in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        

        C_decomp = np.zeros((n, p))
        for r in range(R):
            u_r = u_matrices[r]  
            v_r = v_matrices[r]
            w_r = w_matrices[r]  
            
            uA_sum = np.sum(u_r * A, axis=1, keepdims=True) 
            vB_sum = np.sum(v_r * B, axis=0, keepdims=True) 
            C_decomp += (uA_sum * vB_sum) * w_r  
        
        error = np.linalg.norm(C_true - C_decomp, ord='fro')
        max_error = max(max_error, error)
    
    return max_error


def compute_decomposition_error(u_vectors, v_vectors, w_vectors, n, m, p, tolerance=1e-6):
    return reconstruct_tensor(u_vectors, v_vectors, w_vectors, n, m, p)

