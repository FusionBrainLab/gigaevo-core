
import numpy as np

np.random.seed(44)

from itertools import product
from helper import Data, Matrix, ToddGenerator, NullSpace, BitVec

class Data:
    """Dont change this data class"""
    name: str
    early_decomposition: np.ndarray
    sota_rank: int

def linear_combination_search(mat: Matrix) -> Matrix:
    def random_basis_vector(ns: NullSpace):
        bitvec = BitVec.from_numpy(np.random.binomial(1, 0.6, ns.basis.rows))
        vec = ns.linear_combination(bitvec)
        return vec
    
    def research(mat: Matrix):
        gen = ToddGenerator(mat)
        best_reduction = 0
        best_vector = None
        best_ns = None
        
        for k in range(mat.rows):
            ns: NullSpace = gen.make1(k)
            Y = ns.basis
            vec = random_basis_vector(ns)
            for i in range(min(Y.rows, 1)):
                reduction = ns.rank_divergence(vec ^ Y[i])
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_vector = vec ^ Y[i]
                    best_ns = ns
            
            for l in range(k + 1, mat.rows):
                ns = gen.make2(k, l)
                Y = ns.basis
                vec = random_basis_vector(ns)
                for i in range(min(Y.rows, 1)):
                    reduction = ns.rank_divergence(vec ^ Y[i])
                    if reduction > best_reduction:
                        best_reduction = reduction
                        best_vector = vec ^ Y[i]
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
    res = linear_combination_search(Matrix.from_numpy(context.early_decomposition))
    return res.to_numpy()