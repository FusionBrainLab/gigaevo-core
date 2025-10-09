#example of greedy algorithm
import jax
import jax.numpy as jnp
import optax
from jax import lax
from typing import List, Dict, Any
from jax.nn import sigmoid
from helper import Data, reconstruct_from_multi_binary_factors, reconstruct_from_single_binary_factor, get_residual_num


## EVOLVE - BLOCK - START
def _sample_rank1(key, d: int):
    """List of D binary vectors, one per mode; ensure each is nonzero."""
    ks = jax.random.split(key, 1)
    v = jax.random.bernoulli(ks[0], 0.5, (d,)).astype(jnp.float32)
    idx = jax.random.randint(ks[0], (), 0, d)
    v = jnp.where(v.sum() == 0, v.at[idx].set(1.), v)
    return ks[0], v

def _choose_best(key, residual, samples: int):
    """Sample several candidates; keep the one minimizing residual metric."""
    best_score, best = jnp.inf, None
    for _ in range(samples):
        key, cand = _sample_rank1(key, residual.shape[0])
        sc = get_residual_num(residual, reconstruct_from_single_binary_factor(cand))
        if float(sc) < float(best_score):
            best_score, best = sc, cand
    return key, best, float(best_score)

def search_min_rank(T: jnp.ndarray, samples=128, max_rank=64, tol=1e-6, seed=0) -> Dict[str, Any]:
    key = jax.random.PRNGKey(seed)
    R = jnp.array(T)
    decomposed: List[List[jnp.ndarray]] = []
    cur = 1
    # cur = 1
    for _ in range(max_rank):
        key, best, sc = _choose_best(key, R, samples)
        if best is None or cur < tol: break
        R = R - reconstruct_from_single_binary_factor(best)
        decomposed.append(best); cur = sc
        if cur < tol: break
    return {"rank": len(decomposed), "residual": get_residual_num(R), "factors": jnp.array(decomposed, dtype=jnp.uint8).T}

def get_parametes_based_on_context_data(data: Data, seed: int):
    return {"samples": 20, "max_rank": data.sota_rank, "tol": 1e-6, "seed":seed+1}

## EVOLVE - BLOCK - END
def entrypoint(context: List[Data]) -> List[Dict[str, Any]]:

    res = []
    for i, data in enumerate(context):
        res.append(search_min_rank(T=data.tensor, **get_parametes_based_on_context_data(data, seed=i+1)))
    return res
