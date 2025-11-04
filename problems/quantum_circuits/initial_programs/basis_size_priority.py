
import numpy as np
from helper import Data, Matrix, ToddGenerator, NullSpace

class Data:
    """Dont change this data class"""
    name: str
    early_decomposition: np.ndarray
    sota_rank: int

def bigger_basis_todd(mat: Matrix) -> Matrix:
    def research(mat: Matrix):
        gen = ToddGenerator(mat)
        # Group by basis size and find best in each group
        candidates_by_basis_size = {}
        
        for k in range(mat.rows):
            ns = gen.make1(k)
            Y = ns.basis
            for i in range(Y.rows):
                reduction = ns.rank_divergence(Y[i])
                if reduction > 0:
                    basis_size = Y.rows
                    current_best = candidates_by_basis_size.get(basis_size, (0, None, None))
                    if reduction > current_best[0]:
                        candidates_by_basis_size[basis_size] = (reduction, ns, Y[i])
            
            for l in range(k + 1, mat.rows):
                ns = gen.make2(k, l)
                Y = ns.basis
                for i in range(Y.rows):
                    reduction = ns.rank_divergence(Y[i])
                    if reduction > 0:
                        basis_size = Y.rows
                        current_best = candidates_by_basis_size.get(basis_size, (0, None, None))
                        if reduction > current_best[0]:
                            candidates_by_basis_size[basis_size] = (reduction, ns, Y[i])
        
        # Try smallest basis sizes first (more specific transformations)
        if candidates_by_basis_size:
            for basis_size in sorted(candidates_by_basis_size.keys()):
                reduction, ns, vector = candidates_by_basis_size[basis_size]
                if reduction > 0:
                    return ns.apply(vector)
        
        return mat

    old_rank = mat.rows + 1
    rank = mat.rows
    while rank < old_rank:
        mat = research(mat)
        old_rank = rank
        rank = mat.rows
    return mat


def entrypoint(context: Data) -> np.ndarray:
    res = bigger_basis_todd(Matrix.from_numpy(context.early_decomposition))
    return res.to_numpy()
