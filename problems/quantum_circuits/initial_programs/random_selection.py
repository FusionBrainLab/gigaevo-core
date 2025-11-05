import random
import numpy as np
np.random.seed(44)
from helper import Matrix, ToddGenerator, NullSpace, BitVec

class Data:
    """Dont change this data class"""
    name: str
    early_decomposition: np.ndarray
    sota_rank: int

def random_todd(mat: Matrix) -> Matrix:
    def research(mat: Matrix):
        gen = ToddGenerator(mat)
        candidates = []
        weights = []
        
        for k in range(mat.rows):
            ns = gen.make1(k)
            Y = ns.basis
            for i in range(Y.rows):
                reduction = ns.rank_divergence(Y[i])
                if reduction > 0:
                    candidates.append((ns, Y[i]))
                    weights.append(reduction)
            
            for l in range(k + 1, mat.rows):
                ns = gen.make2(k, l)
                Y = ns.basis
                for i in range(Y.rows):
                    reduction = ns.rank_divergence(Y[i])
                    if reduction > 0:
                        candidates.append((ns, Y[i]))
                        weights.append(reduction)
        
        if candidates:
            total_weight = sum(weights)
            if total_weight > 0:
                r = np.random.uniform(0, total_weight)
                current = 0
                for i, weight in enumerate(weights):
                    current += weight
                    if r <= current:
                        ns, vector = candidates[i]
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
    res = random_todd(Matrix.from_numpy(context.early_decomposition))
    return res.to_numpy()
