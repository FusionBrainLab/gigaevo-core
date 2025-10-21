import jax
import jax.numpy as jnp
import optax
from jax.nn import sigmoid
from jax import lax
from typing import Tuple, List, Any, Dict
from dataclasses import dataclass
from helper import ParityMatrix

@dataclass
class Data:
    name: str
    early_decomposition: ParityMatrix
    sota_rank: int
    
# EVOLVE-BLOCK-START
def smooth_reconstruction(f: jnp.ndarray) -> jnp.ndarray:
    p = sigmoid(f)
    and_r = jnp.einsum("ar,br,cr->abcr", p, p, p)
    return 0.5 * (1.0 - jnp.prod(1.0 - 2.0 * and_r, axis=-1))

def bce_loss_from_probs(t: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
    eps = 1e-6
    P = jnp.clip(P, eps, 1.0 - eps)
    return jnp.sum(-(t * jnp.log(P) + (1.0 - t) * jnp.log1p(-P)))

def to_binary_factors(f: jnp.ndarray) -> jnp.ndarray:
    return (sigmoid(f) >= 0.5).astype(bool)

def logits_from_binary(B: jnp.ndarray, eps: float = 0.05) -> jnp.ndarray:
    probs = eps + (1.0 - 2.0 * eps) * B.astype(jnp.float32)
    return jnp.log(probs) - jnp.log1p(-probs)

def get_optimizer(lr):
    return optax.adam(lr)

def make_trainer(target_T: jnp.ndarray, lr: float):
    opt = get_optimizer(lr)
    def loss_fn(f: jnp.ndarray) -> jnp.ndarray:
        return bce_loss_from_probs(target_T, smooth_reconstruction(f))
    @jax.jit
    def step(f: jnp.ndarray, state: optax.OptState):
        grads = jax.grad(loss_fn)(f)
        updates, state = opt.update(grads, state, f)
        f = optax.apply_updates(f, updates)
        return f, state
    @jax.jit
    def run_steps(f_init: jnp.ndarray, steps: int):
        state = opt.init(f_init)
        def body(i, carry):
            f, s = carry
            f, s = step(f, s)
            return f, s
        f_final, _ = lax.fori_loop(0, steps, body, (f_init, state))
        return f_final, steps
    return run_steps

def perturbate(init: jnp.ndarray, base_key, seed, sigma=5.0):
    init = logits_from_binary(init)
    shape = init.shape[0], init.shape[1] - 1
    key = jax.random.fold_in(base_key, (int(shape[-1]) << 1) + int(seed))
    return init[:,:-1] + jax.random.normal(key, shape) * sigma

def search_min_rank(
                    PME: ParityMatrix, 
                    per_rank_steps: int = 1000, 
                    lr: float = 3e-2, 
                    restarts: int = 1, 
                    seed: int = 42) -> jnp.ndarray:
    T = jnp.asarray(PME.get_target_tensor(), dtype=jnp.uint8)
    PME_B = jnp.asarray(PME.P, dtype=jnp.uint8)
    base_key = jax.random.PRNGKey(seed)
    run_steps = make_trainer(T.astype(jnp.float32), lr)
    for i in range(restarts):
        F0 = perturbate(PME_B, base_key=base_key, seed=seed)
        F, _ = run_steps(F0, per_rank_steps)
        B = to_binary_factors(F)
        new_P = ParityMatrix(B)
        if jnp.all(new_P.T == T): 
            PME_B = new_P
    return ParityMatrix(PME_B)

def get_parameters_based_on_context_data(d: Data, seed: int) -> Dict[str, Any]:
    return {"per_rank_steps": 500, "lr": 6e-2, "restarts": 10, "seed": seed}
# EVOLVE-BLOCK-END

def entrypoint(context: Data) -> ParityMatrix:
    data = context
    params = get_parameters_based_on_context_data(data, seed=42)
    res = search_min_rank(PME=data.early_decomposition, **params)
    return res