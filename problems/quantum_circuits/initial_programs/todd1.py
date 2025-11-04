
import numpy as np
from helper import Data, Matrix, ToddGenerator, NullSpace

class Data:
    """Dont change this data class"""
    name: str
    early_decomposition: np.ndarray
    sota_rank: int

def fast_todd(mat: Matrix) -> Matrix:
    def research(mat: Matrix):
        gen = ToddGenerator(mat)
        for k in range(mat.rows):
            ns = gen.make1(k)
            Y = ns.basis
            for i in range(Y.rows):
                if ns.rank_divergence(Y[i]) > 0:
                    mat = ns.apply(Y[i])
                    return mat
            for l in range(k + 1, mat.rows):
                ns = gen.make2(k, l)
                Y = ns.basis
                for i in range(Y.rows):
                    if ns.rank_divergence(Y[i]) > 0:
                        mat = ns.apply(Y[i])
                        return mat
        return mat
    old_rank = mat.rows + 1
    rank = mat.rows
    while rank < old_rank:
        mat = research(mat)
        old_rank = rank
        rank = mat.rows
    return mat

def entrypoint(context: Data) -> np.ndarray:
    res = fast_todd(Matrix.from_numpy(context.early_decomposition))
    return res.to_numpy()