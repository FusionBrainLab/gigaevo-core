from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jax
from jax import lax
from jax.nn import sigmoid
import jax.numpy as jnp
import optax  # Here is all the optimizer to use


@dataclass
class Data:
    name: str
    tensor: jnp.ndarray
    sota_rank: int


def reconstruct_from_single_binary_factor(f: jnp.ndarray) -> jnp.ndarray:
    f = f.astype(jnp.uint8)
    return jnp.einsum("a,b,c->abc", *(f, f, f)).astype(jnp.uint8)


def reconstruct_from_multi_binary_factors(b: jnp.ndarray) -> jnp.ndarray:
    spec = "ar,br,cr->abcr"
    and_per_r = jnp.einsum(spec, b, b, b).astype(jnp.uint8)
    return (jnp.sum(and_per_r, axis=-1) & jnp.uint8(1)).astype(jnp.uint8)


def get_residual_num(T1: jnp.ndarray, T2: jnp.ndarray = None):
    if T2 is None:
        return int(jnp.sum(T1))
    return jnp.sum(T1 ^ T2)


# EVOLVE-BLOCK-START
def smooth_reconstruction(factors: jnp.ndarray) -> jnp.ndarray:
    p = sigmoid(factors)
    and_per_r = jnp.einsum("ar,br,cr->abcr", p, p, p)
    return 0.5 * (1.0 - jnp.prod(1.0 - 2.0 * and_per_r, axis=-1))


def bce_loss_from_probs(target: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
    eps = 1e-6
    P = jnp.clip(P, eps, 1.0 - eps)
    return jnp.sum(-(target * jnp.log(P) + (1.0 - target) * jnp.log1p(-P)))


def to_binary_factors(factors: jnp.ndarray) -> jnp.ndarray:
    return (sigmoid(factors) >= 0.5).astype(jnp.uint8)


def generate_finit(shape, base_key, seed: int):
    sigma_coef = 10.0
    key = jax.random.fold_in(base_key, (shape[-1] << 1) + seed)
    return jax.random.normal(key, shape) * sigma_coef


def get_optimizer(lr):
    return optax.adam(lr)


def make_trainer(target: jnp.ndarray, lr: float):
    opt = get_optimizer(lr)

    def loss_fn(f: jnp.ndarray) -> jnp.ndarray:
        P = smooth_reconstruction(f)
        return bce_loss_from_probs(target, P)

    @jax.jit
    def step(
        f: jnp.ndarray, state: optax.OptState
    ) -> Tuple[jnp.ndarray, optax.OptState]:
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

        # use jitted loops for step for better speed
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
) -> jnp.array:
    n = T.shape[0]
    base_key = jax.random.PRNGKey(seed)

    best = {"rank": None, "residual": T.size, "factors": None, "steps": 0}

    run_steps = make_trainer(T.astype(jnp.float32), lr)
    for r in range(start, end + 1):
        # multi init for better exploration
        for s in range(restarts):
            F0 = generate_finit((n, r), base_key, seed=s)
            F, used = run_steps(F0, per_rank_steps)
            binF = to_binary_factors(F)
            T_hat = reconstruct_from_multi_binary_factors(binF)
            residual = get_residual_num(T, T_hat)
            if (residual < best["residual"]) or (
                residual == best["residual"]
                and (best["rank"] is None or r < best["rank"])
            ):
                best = {"rank": r, "residual": residual, "factors": binF}
            if residual == 0:
                return best["factors"]
    return best["factors"]


results: List[Dict[str, Any]] = []


def get_parametes_based_on_context_data(T: Data, seed):
    return {
        "per_rank_steps": 500,
        "lr": 6e-2,
        "restarts": 10,
        "seed": seed,
        "start": T.sota_rank - 1,
        "end": T.sota_rank,
    }


def entrypoint(context: List[Data]) -> List[jnp.array]:
    # cylce to find solution for all the problems from the context
    for idx, data in enumerate(context):
        res = search_min_rank(
            T=data.tensor,
            **get_parametes_based_on_context_data(
                data, idx
            ),  # it is gauranteed there exist solution with T.sota_rank
        )
        results.append(res)
    return results


# EVOLVE-BLOCK-END
