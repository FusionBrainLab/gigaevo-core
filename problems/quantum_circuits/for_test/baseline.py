from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax
from jax.nn import sigmoid
import optax
from typing import Tuple, List, Any, Dict
from helper import reconstruct_from_multi_binary_factors, get_residual_num  # ensure JAX impls for speed

DIM = 3

def entrypoint(context: List[jnp.ndarray]) -> List[Dict[str, Any]]:
    def make_spec():
        letters = ''.join(chr(97 + i) for i in range(DIM))
        return ','.join(f"{chr(97+i)}r" for i in range(DIM)) + "->" + letters + "r"

    def smooth_reconstruction(factors: jnp.ndarray, r:int,  spec: str) -> jnp.ndarray:
        p = sigmoid(factors)
        and_per_r = jnp.einsum(spec, *([p] * DIM))
        return 0.5 * (1.0 - jnp.prod(1.0 - 2.0 * and_per_r, axis=-1))

    def bce_loss_from_probs(target: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
        eps = 1e-6
        P = jnp.clip(P, eps, 1.0 - eps)
        return jnp.sum(-(target * jnp.log(P) + (1.0 - target) * jnp.log1p(-P)))

    def to_binary_factors(factors: jnp.ndarray) -> jnp.ndarray:
        return (sigmoid(factors) >= 0.5).astype(jnp.uint8)

    def generate_finit(shape, base_key, seed: int):
        sigma_coef = 10.
        key = jax.random.fold_in(base_key, (shape[-1] << 1) + seed)
        return jax.random.normal(key, shape) * sigma_coef

    def get_optimizer(lr):
        return optax.adam(lr)
    
    def make_trainer(target: jnp.ndarray, spec: str, lr: float):
        opt = get_optimizer(lr)
        def loss_fn(f: jnp.ndarray) -> jnp.ndarray:
            P = smooth_reconstruction(f,  DIM, spec)
            return bce_loss_from_probs(target, P)

        @jax.jit
        def step(f: jnp.ndarray, state: optax.OptState) -> Tuple[jnp.ndarray, optax.OptState]:
            grads = jax.grad(loss_fn)(f)
            updates, state = opt.update(grads, state, f)
            f = optax.apply_updates(f, updates)
            return f, state

        @jax.jit
        def run_steps(f_init: jnp.ndarray, steps: int) -> Tuple[jnp.ndarray, int]:
            state = opt.init(f_init)
            def body(i, carry):
                f, s = carry
                f, s = step(f, s)
                return (f, s)
            f_final, _ = lax.fori_loop(0, steps, body, (f_init, state))
            return f_final, steps
        return run_steps

    def search_min_rank(
        T: jnp.ndarray,
        per_rank_steps: int = 1000,
        lr: float = 3e-2,
        restarts: int = 1,
        seed: int = 1,
        start: int = 10,
        end: int = 11,
        prune_thresh: float = 1e-3,
    ) -> Dict[str, Any]:
        n = T.shape[0]
        spec = make_spec()
        base_key = jax.random.PRNGKey(seed)

        best = {"rank": None, "residual": T.size, "factors": None, "steps": 0}
        steps_used = 0

        run_steps = make_trainer(T.astype(jnp.float32), spec, lr)
        for r in range(start, end + 1):
            for s in range(restarts):
                F0 = generate_finit((n, r), base_key, seed=s)
                F, used = run_steps(F0, per_rank_steps)
                steps_used += int(used)

                # prune nearly-dead columns (cheap, post-step)
                norms = jnp.sqrt(jnp.sum(jnp.square(sigmoid(F) - 0.5), axis=0))
                keep = norms > prune_thresh
                if keep.sum() == 0:
                    continue
                if keep.sum() < r:
                    F = F[:, keep]
                    r = int(F.shape[1])

                binF = to_binary_factors(F)
                T_hat = reconstruct_from_multi_binary_factors(binF, DIM)
                residual = get_residual_num(T, T_hat)
                if (residual < best["residual"]) or (residual == best["residual"] and (best["rank"] is None or r < best["rank"])):
                    best = {"rank": r, "residual": int(residual), "factors": binF, "steps": steps_used}
                if residual == 0:
                    return best
        return best

    results: List[Dict[str, Any]] = []
    for idx, T in enumerate(context):
        
        max_rank = T.sota_rank + 5
        res = search_min_rank(
            T=T.tensor,
            per_rank_steps=5000,
            lr=6e-2,
            restarts=2,
            seed=idx,
            start=max_rank - 1,
            end=max_rank + 3,
        )
        results.append(res)

    return results