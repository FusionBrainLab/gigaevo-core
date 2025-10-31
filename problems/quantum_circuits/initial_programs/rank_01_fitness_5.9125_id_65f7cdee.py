"""
Top Program #1
Program ID: 65f7cdee-8c16-4c5f-bd5d-5d80184a6997
Fitness: 5.9125
Created: 2025-10-11 09:46:29.903397+00:00
Updated: 2025-10-11 09:47:14.343586+00:00
Generation: 7
State: ProgramState.DISCARDED
"""

from dataclasses import dataclass
from typing import List, Tuple

import jax
from jax import lax
from jax.nn import sigmoid
import jax.numpy as jnp
import optax


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
"""Enhanced Waring decomposition with improved temperature annealing and residual updates"""


def smooth_reconstruction(factors: jnp.ndarray) -> jnp.ndarray:
    p = sigmoid(factors)
    and_per_r = jnp.einsum("ar,br,cr->abcr", p, p, p)
    return 0.5 * (1.0 - jnp.prod(1.0 - 2.0 * and_per_r, axis=-1))


def enhanced_bce_loss_with_adaptive_alignment(
    target: jnp.ndarray,
    factors: jnp.ndarray,
    temperature: float = 1.0,
    n: int = 1,
    r: int = 1,
) -> jnp.ndarray:
    """Improved loss with balanced temperature annealing and enhanced binary encouragement"""
    # Balanced temperature annealing
    current_temp = jnp.maximum(temperature, 0.1)
    sharp_factors = factors / current_temp
    p = sigmoid(sharp_factors)

    # Smooth reconstruction with temperature control
    and_per_r = jnp.einsum("ar,br,cr->abcr", p, p, p)
    P_recon = 0.5 * (1.0 - jnp.prod(1.0 - 2.0 * and_per_r, axis=-1))

    # Binary reconstruction for alignment
    binF = (p >= 0.5).astype(jnp.float32)
    T_bin = reconstruct_from_multi_binary_factors(binF.astype(jnp.uint8)).astype(
        jnp.float32
    )

    # Enhanced BCE with error-aware weighting
    error_magnitude = jnp.abs(target - P_recon)
    weight = 1.0 + 4.0 * error_magnitude  # Balanced scaling

    eps = 1e-8
    P_clipped = jnp.clip(P_recon, eps, 1.0 - eps)
    bce_loss = -jnp.sum(
        weight * (target * jnp.log(P_clipped) + (1.0 - target) * jnp.log1p(-P_clipped))
    )

    # Binary alignment loss
    alignment_loss = jnp.sum(jnp.abs(P_recon - T_bin)) * 0.03

    # Enhanced binary encouragement with complexity scaling
    tensor_density = jnp.mean(target)
    # Stronger scaling that persists through convergence
    binary_scale = jnp.minimum(2.0, 15.0 / (n * jnp.sqrt(r))) * (
        1.0 + 0.5 * tensor_density
    )
    binary_encouragement = jnp.mean((p * (1.0 - p)) ** 2) * binary_scale

    return bce_loss + alignment_loss + binary_encouragement


def generate_finit(shape, base_key, seed: int, sparsity: float):
    """Balanced initialization scaling"""
    n, r = shape[0], shape[1]
    key = jax.random.fold_in(base_key, (r << 2) + seed)
    scale_factor = 1.0 / (1.0 + 0.05 * n * jnp.sqrt(r))
    return jax.random.normal(key, shape) * scale_factor


def get_optimizer(lr):
    return optax.adam(lr)


def make_trainer(target: jnp.ndarray, lr: float, sparsity: float, n: int):
    opt = get_optimizer(lr)

    def loss_fn(f: jnp.ndarray, step: int) -> jnp.ndarray:
        # Improved temperature annealing: balanced cooling for better convergence
        r = f.shape[1]
        complexity_factor = jnp.maximum(1.0, n * jnp.sqrt(r) / 25.0)
        # Slower cooling for complex problems, faster for simple ones
        current_temp = 1.0 / jnp.sqrt(step + 100.0 * complexity_factor)
        return enhanced_bce_loss_with_adaptive_alignment(target, f, current_temp, n, r)

    @jax.jit
    def step(
        f: jnp.ndarray, state: optax.OptState, step_count: int
    ) -> Tuple[jnp.ndarray, optax.OptState]:
        grads = jax.grad(loss_fn)(f, step_count)
        updates, state = opt.update(grads, state, f)
        f = optax.apply_updates(f, updates)
        return f, state

    @jax.jit
    def run_steps(f_init: jnp.ndarray, steps: int) -> Tuple[jnp.ndarray, int]:
        state = opt.init(f_init)

        def body(i, carry):
            f, s = carry
            f, s = step(f, s, i)
            return (f, s)

        f_final, _ = lax.fori_loop(0, steps, body, (f_init, state))
        return f_final, steps

    return run_steps


def to_binary_factors(factors: jnp.ndarray) -> jnp.ndarray:
    return (sigmoid(factors) >= 0.5).astype(jnp.uint8)


def warm_start_expansion(
    F_prev: jnp.ndarray, new_rank: int, base_key, seed: int, sparsity: float
) -> jnp.ndarray:
    """Enhanced warm-start with adaptive preservation strategy"""
    n, prev_rank = F_prev.shape
    if prev_rank >= new_rank:
        return F_prev[:, :new_rank]

    binF_prev = to_binary_factors(F_prev)
    T_prev = reconstruct_from_multi_binary_factors(binF_prev)
    individual_contributions = []
    for r in range(prev_rank):
        factor_removed = jnp.concatenate(
            [binF_prev[:, :r], binF_prev[:, r + 1 :]], axis=1
        )
        T_partial = reconstruct_from_multi_binary_factors(factor_removed)
        residual_increase = get_residual_num(T_prev, T_partial)
        individual_contributions.append(residual_increase)

    contributions = jnp.array(individual_contributions)
    if prev_rank > 1:
        threshold = jnp.percentile(contributions, max(20.0, 50.0 - 10.0 * prev_rank))
    else:
        threshold = contributions[0]

    keep_mask = contributions >= threshold

    F_kept = F_prev[:, keep_mask]

    key = jax.random.fold_in(base_key, seed)
    new_factors = generate_finit((n, new_rank - F_kept.shape[1]), key, seed, sparsity)
    return jnp.concatenate([F_kept, new_factors], axis=1)


def adaptive_greedy_backup(T: jnp.ndarray, max_rank: int, seed: int) -> jnp.array:
    """Enhanced greedy decomposition with monotonic residual updates"""
    key = jax.random.PRNGKey(seed)
    R = T.copy()
    factors = []
    n = T.shape[0]

    tensor_density = jnp.sum(T) / (n**3)
    sparsity = max(0.1, min(0.9, 1.0 - tensor_density))

    adaptive_max_rank = max_rank + int(4 * tensor_density)

    for _ in range(adaptive_max_rank):
        residual_sum = get_residual_num(R)
        if residual_sum == 0:
            break

        base_trials = min(80, max(20, n * 3))
        trials = base_trials + min(60, residual_sum // 2)

        best_score = jnp.inf
        best_factor = None

        for trial in range(trials):
            key, subkey = jax.random.split(key)
            candidate = jax.random.bernoulli(subkey, sparsity, (n,)).astype(jnp.uint8)

            if jnp.sum(candidate) == 0:
                idx = jax.random.randint(subkey, (), 0, n)
                candidate = candidate.at[idx].set(1)

            T_cand = reconstruct_from_single_binary_factor(candidate)
            # CRITICAL FIX: Use AND-NOT for monotonic residual reduction
            new_residual = get_residual_num(R & ~T_cand)
            score = new_residual

            if score < best_score:
                best_score, best_factor = score, candidate

            if best_score == 0:
                break

        if best_factor is not None:
            factors.append(best_factor)
            T_cand = reconstruct_from_single_binary_factor(best_factor)
            # CRITICAL FIX: Maintain monotonic residual reduction
            R = (R & ~T_cand).astype(jnp.uint8)

    return (
        jnp.array(factors, dtype=jnp.uint8).T
        if factors
        else jnp.zeros((n, 0), dtype=jnp.uint8)
    )


def adaptive_min_rank_search(
    T: jnp.ndarray,
    sota_rank: int,
    base_steps: int = 600,
    lr: float = 1e-2,
    restarts: int = 8,
    seed: int = 1,
    max_expansion: int = 5,
) -> jnp.array:
    n = T.shape[0]
    base_key = jax.random.PRNGKey(seed)

    density = jnp.sum(T) / (n**3)
    sparsity = 1.0 - density

    adaptive_lr = (
        lr * jnp.minimum(1.0, jnp.sqrt(10.0 / n)) * jnp.sqrt(jnp.maximum(density, 0.05))
    )

    adaptive_expansion = min(max_expansion + 1, max(3, n // 6 + int(3 * density)))
    start_rank = max(1, sota_rank - min(3, sota_rank // 2))
    end_rank = sota_rank + adaptive_expansion
    best = {"rank": None, "residual": T.size, "factors": None}

    run_steps = make_trainer(T.astype(jnp.float32), adaptive_lr, float(density), n)

    best_prev_continuous = None

    for r in range(start_rank, end_rank + 1):
        steps = base_steps + (r - start_rank) * 100 + (n // 4) * 30

        rank_best_residual = T.size
        rank_best_factors = None
        rank_best_continuous = None

        for s in range(restarts):
            if best_prev_continuous is not None and s < restarts * 2 // 3:
                F0 = warm_start_expansion(
                    best_prev_continuous, r, base_key, seed=s, sparsity=sparsity
                )
            else:
                F0 = generate_finit((n, r), base_key, seed=s, sparsity=float(density))

            F, _ = run_steps(F0, steps)
            binF = to_binary_factors(F)
            T_hat = reconstruct_from_multi_binary_factors(binF)
            residual = get_residual_num(T, T_hat)

            if residual < rank_best_residual:
                rank_best_residual = residual
                rank_best_factors = binF
                rank_best_continuous = F

            if residual == 0:
                return binF

        if rank_best_residual < best["residual"] or (
            rank_best_residual == best["residual"]
            and (best["rank"] is None or r < best["rank"])
        ):
            best = {
                "rank": r,
                "residual": rank_best_residual,
                "factors": rank_best_factors,
            }
            best_prev_continuous = rank_best_continuous
            if rank_best_residual == 0:
                return rank_best_factors

    return best["factors"]


def integrated_decomposition_search(
    T: jnp.ndarray,
    sota_rank: int,
    base_steps: int = 600,
    lr: float = 1e-2,
    restarts: int = 8,
    seed: int = 1,
    max_expansion: int = 5,
) -> jnp.array:
    """Enhanced integrated search with improved backup strategy"""
    n = T.shape[0]

    opt_result = adaptive_min_rank_search(
        T=T,
        sota_rank=sota_rank,
        base_steps=base_steps,
        lr=lr,
        restarts=restarts,
        seed=seed,
        max_expansion=max_expansion,
    )

    T_opt = reconstruct_from_multi_binary_factors(opt_result)
    residual_opt = get_residual_num(T, T_opt)

    if residual_opt == 0:
        return opt_result

    greedy_result = adaptive_greedy_backup(
        T, sota_rank + max_expansion + 2, seed + 1000
    )
    T_greedy = reconstruct_from_multi_binary_factors(greedy_result)
    residual_greedy = get_residual_num(T, T_greedy)

    if residual_greedy < residual_opt or (
        residual_greedy == residual_opt and greedy_result.shape[1] < opt_result.shape[1]
    ):
        return greedy_result

    return opt_result


def get_parametes_based_on_context_data(data: Data, seed: int):
    n = data.tensor.shape[0]
    density = jnp.sum(data.tensor) / (n**3)

    restarts = min(12, max(6, int(n * jnp.sqrt(density + 1e-6))))
    base_steps = min(900, max(400, int(n * 32 * jnp.sqrt(1.0 / (density + 1e-6)))))

    max_expansion = min(6, max(3, n // 5 + int(2.5 * density)))

    return {
        "sota_rank": data.sota_rank,
        "base_steps": int(base_steps),
        "lr": 1e-2,
        "restarts": int(restarts),
        "seed": seed,
        "max_expansion": int(max_expansion),
    }


def entrypoint(context: List[Data]) -> List[jnp.array]:
    """Return list of an integer array of shapes (N, R) correspoding to waring decomposition.
    Primary objective: find exact decomposition with minimal rank."""
    results = []
    for idx, data in enumerate(context):
        try:
            res = integrated_decomposition_search(
                T=data.tensor,
                **get_parametes_based_on_context_data(data, idx),
            )
        except Exception:
            res = adaptive_greedy_backup(data.tensor, data.sota_rank + 4, idx)
        results.append(res)
    return results


# EVOLVE-BLOCK-END
