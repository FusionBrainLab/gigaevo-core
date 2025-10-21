from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from helper import ParityMatrix
from dataclasses import dataclass

@dataclass
class Data:
    name: str
    early_decomposition: ParityMatrix
    sota_rank: int

# EVOLVE-BLOCK-START
def get_any_kernel_vector(matrix: np.ndarray, augmented_matrix: Optional[np.ndarray]=None) -> Optional[np.ndarray]:
    """
    Find a kernel vector using Gaussian elimination.
    Args:
        matrix: The coefficient matrix (list of boolean vectors)
    Returns:
        A kernel vector if found, None otherwise
    """
    pivots = {}
    def get_first_one(arr: np.ndarray) -> Optional[int]:
        """Find the index of the first True value in the array."""
        nonzero_indices = np.argwhere(arr)
        return nonzero_indices[0][0] if len(nonzero_indices) > 0 else 0
    
    n = matrix.shape[0]
    if augmented_matrix is None:
        augmented_matrix = np.eye(n, dtype=bool)
    for i in range(n):
        if i in pivots: continue
        for key, value in list(pivots.items()):
            if matrix[i,value]:
                matrix[i,:] ^= matrix[key,:]
                augmented_matrix[i] ^= augmented_matrix[key]
        
        pivot_col = get_first_one(matrix[i])
        
        if matrix[i, pivot_col]:
            for j in list(pivots.keys()):
                if matrix[j][pivot_col]:
                    matrix[j] ^= matrix[i]
                    augmented_matrix[j] ^= augmented_matrix[i]
            
            pivots[i] = pivot_col
        else:
            return augmented_matrix[i].copy()
    return None

def tohpe(PME: ParityMatrix):
    """
    Algorithm for achieving upper bound of decomposition (n^2 + n) //2 + 1
    P += zy^T
    z - any under conditions:
    |y| = 0 -- this is parity
    |P[i,:] and y| = 0            | 
    |P[i,:] and P[j,:] and y| = 0 | this is matrix
    """
    PM = PME
    def choose_best_z(y):
        parity = y.sum() % 2

        e = np.argwhere(y == 1)
        n = np.argwhere(y == 0)
        if e.size != 0 and n.size != 0:
            if parity == 1:
                PM.add_factors(np.array([0]*PM.get_target_qubits(), bool))
            a, b = e[0], n[0]
            return PM.P[:,a[0]] ^ PM.P[:,b[0]], e
        return 0, 0 
    
    nb_qubits = PM.get_target_qubits()
    while True:
        displ = 0
        matrix = np.zeros((nb_qubits * (nb_qubits + 1) // 2, PM.num_factors()))
        matrix[0:nb_qubits, :] = PM.P 
        for i in range(1, nb_qubits):
            displ += nb_qubits - i + 1
            matrix[displ:displ + nb_qubits - i,:] = np.einsum("ar,r->ar", PM.P[i:,:], PM.P[i - 1,:])
        matrix = matrix.T.astype(np.bool)
        y = get_any_kernel_vector(matrix)
        
        z, places = choose_best_z(y)
        if not isinstance(z, int):
            PM.add_to_factors([z]*len(places), places)
            PM.destroy_duplicate_columns()
        else: 
            break
    return PM
# EVOLVE-BLOCK-END

def entrypoint(context: Data) -> ParityMatrix:
    data = context
    res = tohpe(data.early_decomposition)
    return res