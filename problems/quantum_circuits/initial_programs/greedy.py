import jax
import jax.numpy as jnp
from typing import List, Dict, Any
from helper import Data, ParityMatrix

# EVOLVE-BLOCK-START
def random_reduce_with_add(pm: ParityMatrix, samples: int, max_rank: int, seed: int) -> jnp.ndarray:
    key = jax.random.PRNGKey(seed)
    bestP = pm.P.copy()
    bestR = bestP.shape[1]
    n, r = bestP.shape
    for s in range(samples):
        if bestR <= max_rank:
            break
        k1 = int(jax.random.randint(jax.random.fold_in(key, 2*s), (), 0, r))
        k2 = int(jax.random.randint(jax.random.fold_in(key, 2*s+1), (), 0, r))
        m = int(jax.random.randint(jax.random.fold_in(key, 3*s+2), (), 1, min(4, r)+1))
        idxs = set([k1])
        while len(idxs) < m:
            idxs.add(int(jax.random.randint(jax.random.fold_in(key, 5*s+len(idxs)), (), 0, r)))
        idxs = sorted(list(idxs))
        zs = []
        for t, idx in enumerate(idxs):
            if t == 0:
                zs.append(bestP[:, k2].copy())
            else:
                zkey = jax.random.fold_in(key, 7*s+idx)
                z = jax.random.bernoulli(zkey, 0.5, (n,)).astype(jnp.uint8)
                zs.append(z)
        try:
            trial = ParityMatrix(bestP.copy())
            trial.add_to_factors(tuple(zs), tuple(idxs))
            trial.destroy_duplicate_columns()
            rnew = trial.P.shape[1]
            if rnew < bestR:
                bestP = trial.P.copy()
                bestR = rnew
        except Exception:
            continue
    return ParityMatrix(bestP)

def search_min_rank(
                    PME: ParityMatrix,
                    samples: int, max_rank: int, seed: int) -> jnp.ndarray:
    return random_reduce_with_add(PME, samples=samples, max_rank=max_rank, seed=seed)

def get_parametes_based_on_context_data(data: Data, seed: int):
    return {"samples": 200, "max_rank": data.sota_rank, "tol": 1e-6, "seed": seed+1}
# EVOLVE-BLOCK-END

def entrypoint(context: Data) -> jnp.array:
    data = context
    params = get_parametes_based_on_context_data(data, seed=1)
    res = search_min_rank(data.early_decomposition, samples=params["samples"], max_rank=params["max_rank"], seed=params["seed"])
    return res
