import jax
import jax.numpy as jnp
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Data:
    name: str
    tensor: jnp.ndarray
    sota_rank: int

def reconstruct_from_single_binary_factor(f: jnp.ndarray) -> jnp.ndarray:
    f = f.astype(jnp.uint8)
    return jnp.einsum("a,b,c->abc", *(f,f,f)).astype(jnp.uint8)


def reconstruct_from_multi_binary_factors(b: jnp.ndarray) -> jnp.ndarray:
    spec = "ar,br,cr->abcr"
    and_per_r = jnp.einsum(spec, b,b,b).astype(jnp.uint8)
    return (jnp.sum(and_per_r, axis=-1) & jnp.uint8(1)).astype(jnp.uint8)

def get_residual_num(T1: jnp.ndarray, T2: jnp.ndarray=None):
    if T2 is None:
        return int(jnp.sum(T1))
    return jnp.sum(T1 ^ T2)

# EVOLVE-BLOCK-START
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
    for _ in range(max_rank):
        key, best, sc = _choose_best(key, R, samples)
        if best is None or cur < tol: break
        R = R - reconstruct_from_single_binary_factor(best)
        decomposed.append(best); cur = sc
        if cur < tol: break
    return jnp.array(decomposed, dtype=jnp.uint8).T

def get_parametes_based_on_context_data(data: Data, seed: int):
    return {"samples": 20, "max_rank": data.sota_rank, "tol": 1e-6, "seed":seed+1}
# EVOLVE-BLOCK-END


def entrypoint(context: List[Data]) -> List[Dict[str, Any]]:
    res = []
    for i, data in enumerate(context):
        res.append(search_min_rank(T=data.tensor, **get_parametes_based_on_context_data(data, seed=i+1)))
    return res

# You are not perrmitted to change this code. This function is called from outside of the program
def evaluate(
    payload: tuple[list[Data], list[jnp.ndarray]],
    W_EXACT: float = 10.0,      # bonus for residual==0
    W_UNDER_SOTA: float = 50.0,   # extra bonus if rank < sota
    W_AT_SOTA: float = 10.0,       # blonus if rank == sota
) -> dict[str, float]:
    context, result = payload
    score = 0
    exact_num = 0
    under_sota_num = 0 
    for con, res in zip(context, result):
        T_rec = reconstruct_from_multi_binary_factors(res)
        residual = get_residual_num(con.tensor, T_rec)
        rank = res.shape[-1]
        score += 5*(1 - residual/T_rec.size)
        if residual == 0:
            exact_num = +1
            score += W_EXACT
            if rank == con.sota_rank:
                score += W_AT_SOTA
            if rank < con.sota_rank:
                under_sota_num += 1
                score += W_UNDER_SOTA
    return {"fitness": score, "exact_num": exact_num, "under_sota": under_sota_num, "is_valid": 1}