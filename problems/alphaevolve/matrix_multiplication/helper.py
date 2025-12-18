import numpy as np


def compute_matrix_multiplication_tensor(m, n, p):
    return {"m": m, "n": n, "p": p}


def reconstruct_tensor(u_vectors, v_vectors, w_vectors, m, n, p):
    R = u_vectors.shape[0]
    

    u_matrices = u_vectors.reshape(R, m, n) 
    v_matrices = v_vectors.reshape(R, n, p)  
    w_matrices = w_vectors.reshape(R, m, p) 
    
    np.random.seed(42)
    max_error = 0.0
    
    for test_idx in range(10):
        A = np.random.randn(m, n)
        B = np.random.randn(n, p)
        C_true = A @ B
        

        C_decomp = np.zeros((m, p))
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


def compute_decomposition_error(u_vectors, v_vectors, w_vectors, m, n, p, tolerance=1e-6):
    return reconstruct_tensor(u_vectors, v_vectors, w_vectors, m, n, p)


def get_target_rank(m, n, p):
    targets = {
        (2, 4, 5): 33,
        (2, 4, 7): 46,
        (2, 4, 8): 52,
        (2, 5, 6): 23,
        (3, 3, 3): 56,
        (3, 4, 6): 66,
        (3, 4, 7): 75,
        (3, 4, 8): 70,
        (3, 5, 6): 82,
        (4, 4, 4): 49,  
        (4, 4, 5): 62,
        (4, 4, 7): 87,
        (4, 4, 8): 98,
        (4, 5, 6): 93,
        (5, 5, 5): 93,
    }
    
    key = (m, n, p)
    if key in targets:
        return targets[key]
    
    return m * n * p

