"""
Top Program #3
Program ID: 7cda9bb1-c89a-4e49-bb78-89ad535e6105
Fitness: 5.5664
Created: 2025-10-10 17:35:22.410127+00:00
Updated: 2025-10-10 17:40:01.600701+00:00
Generation: 5
State: ProgramState.EVOLVING
"""

import jax
import jax.numpy as jnp
import optax 
from jax.nn import sigmoid
from jax import lax
from dataclasses import dataclass
from typing import Tuple, List, Any, Dict

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
"""Enhanced Waring decomposition with optimized temperature annealing and improved warm-start strategy"""
def smooth_reconstruction(factors: jnp.ndarray) -> jnp.ndarray:
    p = sigmoid(factors)
    and_per_r = jnp.einsum("ar,br,cr->abcr", p,p,p)
    return 0.5 * (1.0 - jnp.prod(1.0 - 2.0 * and_per_r, axis=-1))

def enhanced_bce_loss_with_alignment(target: jnp.ndarray, factors: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """Combined loss with optimized temperature annealing and binary alignment"""
    # Optimized temperature annealing: balanced cooling for exploration vs convergence
    sharp_factors = factors / jnp.maximum(temperature, 0.01)
    p = sigmoid(sharp_factors)
    
    # Smooth reconstruction with temperature control
    and_per_r = jnp.einsum("ar,br,cr->abcr", p, p, p)
    P_recon = 0.5 * (1.0 - jnp.prod(1.0 - 2.0 * and_per_r, axis=-1))
    
    # Binary reconstruction for alignment
    binF = (p >= 0.5).astype(jnp.float32)
    T_bin = reconstruct_from_multi_binary_factors(binF.astype(jnp.uint8)).astype(jnp.float32)
    
    # Enhanced BCE with error-aware weighting
    error_magnitude = jnp.abs(target - P_recon)
    weight = 1.0 + 5.0 * error_magnitude  # Moderate scaling
    
    eps = 1e-8
    P_clipped = jnp.clip(P_recon, eps, 1.0 - eps)
    bce_loss = -jnp.sum(weight * (target * jnp.log(P_clipped) + (1.0 - target) * jnp.log1p(-P_clipped)))
    
    # Binary alignment loss - bridges probability/binary gap (beneficial from Parent 2)
    alignment_loss = jnp.sum(jnp.abs(P_recon - T_bin)) * 0.05
    
    # Improved binary encouragement with better scaling
    n, r = factors.shape
    binary_scale = jnp.minimum(1.0, 15.0 / (n * jnp.sqrt(r)))  # Balanced scaling from Parent 2
    binary_encouragement = jnp.mean((p * (1.0 - p)) ** 2) * binary_scale
    
    return bce_loss + alignment_loss + binary_encouragement

def generate_finit(shape, base_key, seed: int, sparsity: float):
    """Improved initialization scaling"""
    n, r = shape[0], shape[1]
    key = jax.random.fold_in(base_key, (r << 2) + seed)
    # Balanced scaling for better exploration
    scale_factor = 1.5 / jnp.sqrt(n * jnp.sqrt(r))
    return jax.random.normal(key, shape) * scale_factor

def get_optimizer(lr):
    return optax.adam(lr)

def make_trainer(target: jnp.ndarray, lr: float, sparsity: float):
    opt = get_optimizer(lr)
    
    def loss_fn(f: jnp.ndarray, step: int) -> jnp.ndarray:
        # Optimized temperature annealing: gradual cooling for better convergence
        current_temp = jnp.maximum(1.0 / jnp.sqrt(step + 200.0), 0.02)  # Balanced cooling
        return enhanced_bce_loss_with_alignment(target, f, current_temp)

    @jax.jit
    def step(f: jnp.ndarray, state: optax.OptState, step_count: int) -> Tuple[jnp.ndarray, optax.OptState]:
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

def warm_start_expansion(F_prev: jnp.ndarray, new_rank: int, base_key, seed: int, sparsity: float) -> jnp.ndarray:
    """Enhanced warm-start with selective factor preservation"""
    n, prev_rank = F_prev.shape
    if prev_rank >= new_rank:
        return F_prev[:, :new_rank]
    
    # Enhanced warm-start: preserve more factors for better convergence
    binF_prev = to_binary_factors(F_prev)
    T_prev = reconstruct_from_multi_binary_factors(binF_prev)
    individual_contributions = []
    for r in range(prev_rank):
        factor_removed = jnp.concatenate([binF_prev[:, :r], binF_prev[:, r+1:]], axis=1)
        T_partial = reconstruct_from_multi_binary_factors(factor_removed)
        residual_increase = get_residual_num(T_prev, T_partial)
        individual_contributions.append(residual_increase)
    
    # Keep factors with meaningful contributions (above 25% of max)
    contributions = jnp.array(individual_contributions)
    threshold = jnp.max(contributions) * 0.25  # More conservative threshold
    keep_mask = contributions > threshold
    
    F_kept = F_prev[:, keep_mask]
    
    key = jax.random.fold_in(base_key, seed)
    new_factors = generate_finit((n, new_rank - F_kept.shape[1]), key, seed, sparsity)
    return jnp.concatenate([F_kept, new_factors], axis=1)

def adaptive_greedy_backup(T: jnp.ndarray, max_rank: int, seed: int) -> jnp.array:
    """Enhanced greedy decomposition with sparsity-aware initialization"""
    key = jax.random.PRNGKey(seed)
    R = T.copy()
    factors = []
    n = T.shape[0]
    
    # Adaptive sparsity based on tensor density
    tensor_density = jnp.sum(T) / (n ** 3)
    sparsity = max(0.1, min(0.9, 1.0 - tensor_density))  # Sparsity opposite to density
    
    # Adaptive max rank based on tensor complexity
    adaptive_max_rank = max_rank + int(6 * tensor_density)  # Balanced expansion
    
    for _ in range(adaptive_max_rank):
        residual_sum = get_residual_num(R)
        if residual_sum == 0:
            break
            
        # Enhanced exploration with density-aware trials
        base_trials = min(96, max(24, n * 4))
        trials = base_trials + min(80, residual_sum // 3)  # More trials for larger residuals
        
        best_score = jnp.inf
        best_factor = None
        
        for trial in range(trials):
            key, subkey = jax.random.split(key)
            candidate = jax.random.bernoulli(subkey, sparsity, (n,)).astype(jnp.uint8)  # Sparsity-aware
            
            if jnp.sum(candidate) == 0:
                idx = jax.random.randint(subkey, (), 0, n)
                candidate = candidate.at[idx].set(1)
            
            T_cand = reconstruct_from_single_binary_factor(candidate)
            new_residual = get_residual_num(R, T_cand)
            score = new_residual
            
            if score < best_score:
                best_score, best_factor = score, candidate
                
            if best_score == 0:
                break
        
        if best_factor is not None:
            factors.append(best_factor)
            T_cand = reconstruct_from_single_binary_factor(best_factor)
            R = (R ^ T_cand).astype(jnp.uint8)
    
    return jnp.array(factors, dtype=jnp.uint8).T if factors else jnp.zeros((n, 0), dtype=jnp.uint8)

def adaptive_min_rank_search(
    T: jnp.ndarray,
    sota_rank: int,
    base_steps: int = 600,
    lr: float = 1e-2,
    restarts: int = 8,
    seed: int = 1,
    max_expansion: int = 6,
) -> jnp.array:
    n = T.shape[0]
    base_key = jax.random.PRNGKey(seed)
    
    # Compute tensor density for adaptive parameters
    density = jnp.sum(T) / (n ** 3)
    sparsity = 1.0 - density
    
    # Enhanced adaptive learning rate
    adaptive_lr = lr * jnp.minimum(1.0, jnp.sqrt(12.0 / n)) * jnp.sqrt(density + 1e-6)
    
    # Balanced expansion for complex tensors
    adaptive_expansion = min(max_expansion + 2, max(4, n // 6 + int(4 * density)))
    start_rank = max(1, sota_rank - min(3, sota_rank // 2))
    end_rank = sota_rank + adaptive_expansion
    best = {"rank": None, "residual": T.size, "factors": None}
    
    run_steps = make_trainer(T.astype(jnp.float32), adaptive_lr, float(density))
    
    # Enhanced warm-start tracking
    best_prev_continuous = None
    
    for r in range(start_rank, end_rank + 1):
        # Balanced step scaling
        steps = base_steps + (r - start_rank) * 120 + (n // 4) * 40
        
        rank_best_residual = T.size
        rank_best_factors = None
        rank_best_continuous = None
        
        for s in range(restarts):
            # Use selective warm-start when available
            if best_prev_continuous is not None and s < restarts // 2:
                F0 = warm_start_expansion(best_prev_continuous, r, base_key, seed=s, sparsity=sparsity)
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
        
        # Update global best with improved criteria
        if rank_best_residual < best["residual"] or (rank_best_residual == best["residual"] and (best["rank"] is None or r < best["rank"])):
            best = {"rank": r, "residual": rank_best_residual, "factors": rank_best_factors}
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
    max_expansion: int = 6,
) -> jnp.array:
    """Enhanced integrated search with conditional greedy backup"""
    n = T.shape[0]
    
    # First attempt with enhanced optimization approach
    opt_result = adaptive_min_rank_search(
        T=T, sota_rank=sota_rank, base_steps=base_steps, 
        lr=lr, restarts=restarts, seed=seed, max_expansion=max_expansion
    )
    
    # Check if optimization found exact solution
    T_opt = reconstruct_from_multi_binary_factors(opt_result)
    residual_opt = get_residual_num(T, T_opt)
    
    if residual_opt == 0:
        return opt_result
    
    # Only run greedy backup if residual is moderate
    if residual_opt > max(5, n // 4):  # Balanced threshold
        greedy_result = adaptive_greedy_backup(T, sota_rank + max_expansion + 2, seed + 1000)
        T_greedy = reconstruct_from_multi_binary_factors(greedy_result)
        residual_greedy = get_residual_num(T, T_greedy)
        
        # Choose decomposition with minimal rank for equal residuals
        if residual_greedy < residual_opt or (residual_greedy == residual_opt and greedy_result.shape[1] < opt_result.shape[1]):
            return greedy_result
    
    return opt_result

def get_parametes_based_on_context_data(data: Data, seed: int):
    n = data.tensor.shape[0]
    density = jnp.sum(data.tensor) / (n ** 3)
    
    # Enhanced adaptive parameters
    restarts = min(14, max(6, int(n * 1.2 * jnp.sqrt(density + 1e-6))))
    base_steps = min(1000, max(500, int(n * 35 * jnp.sqrt(1.0 / (density + 1e-6)))))
    
    # Balanced expansion for dense tensors
    max_expansion = min(7, max(4, n // 5 + int(3 * density)))
    
    return {
        "sota_rank": data.sota_rank,
        "base_steps": int(base_steps),
        "lr": 1e-2,
        "restarts": int(restarts),
        "seed": seed,
        "max_expansion": int(max_expansion)
    }

def entrypoint(context: List[Data]) -> List[jnp.array]:
    """Return list of an integer array of shapes (N, R) correspoding to waring decomposition.
    Primary objective: find exact decomposition with minimal rank."""
    results = []
    for idx, data in enumerate(context):
        try:
            # Use enhanced integrated decomposition search
            res = integrated_decomposition_search(
                T=data.tensor,
                **get_parametes_based_on_context_data(data, idx),
            )
        except Exception:
            # Fallback to adaptive greedy approach
            res = adaptive_greedy_backup(data.tensor, data.sota_rank + 4, idx)
        results.append(res)
    return results
# EVOLVE-BLOCK-END
