
import numpy as np
from helper import Data, Matrix, ToddGenerator, NullSpace

class Data:
    """Dont change this data class"""
    name: str
    early_decomposition: Matrix
    sota_rank: int

def greedy_todd(mat: Matrix) -> Matrix:
    def research(mat: Matrix):
        gen = ToddGenerator(mat)
        best_reduction = 0
        best_vector = None
        best_ns = None
        
        for k in range(mat.rows):
            # Try single index transformations
            ns: NullSpace = gen.make1(k)
            Y = ns.basis
            for i in range(min(Y.rows, 1)):
                reduction = ns.rank_divergence(Y[i])
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_vector = Y[i]
                    best_ns = ns
            
            for l in range(k + 1, mat.rows):
                ns = gen.make2(k, l)
                Y = ns.basis
                for i in range(min(Y.rows, 1)):
                    reduction = ns.rank_divergence(Y[i])
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_vector = Y[i]
                        best_ns = ns
        
        if best_vector is not None:
            return best_ns.apply(best_vector)
        return mat
    old_rank = mat.rows + 1
    rank = mat.rows
    while rank < old_rank:
        mat = research(mat)
        old_rank = rank
        rank = mat.rows
    return mat

def entrypoint(context: Data) -> np.ndarray:
    res = greedy_todd(Matrix.from_numpy(context.early_decomposition))
    return res.to_numpy()