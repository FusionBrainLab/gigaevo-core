import numpy as np
import jax.numpy as jnp
from typing import List, Dict, Any
from dataclasses import dataclass
from helper import ParityMatrix

@dataclass
class Data:
    name: str
    early_decomposition: ParityMatrix
    sota_rank: int

# EVOLVE-BLOCK-START
def random_reduce_with_add(pm: ParityMatrix, samples: int, max_rank: int, seed: int) -> ParityMatrix:
    rng = np.random.default_rng(seed)
    bestP = pm.P.copy().astype(np.uint8)
    for s in range(samples):
        n, r = bestP.shape
        if r == 0 or r <= max_rank: break
        k1 = int(rng.integers(0, r))
        k2 = int(rng.integers(0, r))
        m = int(rng.integers(1, min(4, r) + 1))
        if m > 1:
            pool = np.delete(np.arange(r), k1)
            others = rng.choice(pool, size=m-1, replace=False)
            idxs = np.sort(np.concatenate([[k1], others])).astype(int)
        else:
            idxs = np.array([k1], dtype=int)
        Z = rng.integers(0, 2, size=(n, m), dtype=np.uint8)
        Z[:, 0] = bestP[:, k2]
        try:
            trial = ParityMatrix(bestP.copy())
            trial.add_to_factors(tuple(Z.T.copy()), tuple(np.array([i], dtype=int) for i in idxs))
            trial.destroy_duplicate_columns()
            if trial.P.shape[1] < r: bestP = trial.P.copy()
        except Exception:
            pass
    return ParityMatrix(bestP)

def search_min_rank(
                    PME: ParityMatrix,
                    samples: int, max_rank: int, seed: int) -> np.ndarray:
    return random_reduce_with_add(PME, samples=samples, max_rank=max_rank, seed=seed)

def get_parametes_based_on_context_data(data: Data, seed: int):
    return {"samples": 5, "max_rank": data.sota_rank, "tol": 1e-6, "seed": seed+1}
# EVOLVE-BLOCK-END

def entrypoint(context: Data) -> ParityMatrix:
    data = context
    params = get_parametes_based_on_context_data(data, seed=1)
    res = search_min_rank(data.early_decomposition, samples=params["samples"], max_rank=params["max_rank"], seed=params["seed"])
    return res
