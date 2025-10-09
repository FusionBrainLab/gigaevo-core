# Insight (algebraic): Exploit symmetry by precomputing tensor slices.
# Insight (optimization): Adaptive learning rate based on residual change.
# Lineage (imitation): Increased rank proportionally to tensor size, as successful in prior runs.
# Design: Precompute tensor slices for faster residual calculation; adaptive learning rate; rank initialization based on tensor size.
import jax
import jax.numpy as jnp
from jax import jit
import optax
from typing import List, Dict
from helper import Data, reconstruct_from_multi_binary_factors, reconstruct_from_single_binary_factor, get_residual_num

def sigmoid(x):
  return 1.0 / (1.0 + jnp.exp(-x))

def binary_cross_entropy_with_logits(logits, labels):
  return jnp.mean(jnp.maximum(logits, 0) - logits * labels + jnp.log(1 + jnp.exp(-jnp.abs(logits))))

@jit
def loss_fn(params, T, precomputed_slices):
    n, r = params.shape
    factors_binary = sigmoid(params) > 0.5
    reconstructed_tensor = jnp.zeros_like(T, dtype=jnp.bool_)
    for i in range(r):
      factor = factors_binary[:, i]
      reconstructed_tensor = jnp.bitwise_xor(reconstructed_tensor, reconstruct_from_single_binary_factor(factor))
    residual = jnp.sum(jnp.bitwise_xor(T, reconstructed_tensor))
    loss = residual + 0.01 * jnp.sum(jnp.abs(params))
    return loss.astype(jnp.float32)

def precompute_tensor_slices(T):
    n = T.shape[0]
    slices = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                slices.append(T[i, j, k])
    return jnp.array(slices)

@jit
def train_step(params, T, optimizer_state, precomputed_slices):
    loss, grads = jax.value_and_grad(loss_fn)(params, T, precomputed_slices)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss

def entrypoint(context: List[Data]) -> List[Dict]:
    results = []
    for data in context:
        T = data.tensor
        n = T.shape[0]
        sota_rank = data.sota_rank

        best_rank = sota_rank * 2
        best_residual = float('inf')
        best_factors = None

        for rank in range(sota_rank - 5, sota_rank + 5): # # dont use start = 1, as it lead to too long computations
            params = jax.random.normal(jax.random.PRNGKey(0), (n, rank)) * 0.1
            precomputed_slices = precompute_tensor_slices(T)

            global optimizer
            optimizer = optax.adam(learning_rate=0.1)
            optimizer_state = optimizer.init(params)

            prev_residual = float('inf')
            residual_tolerance = 1e-4
            plateau_count = 0
            max_plateau = 10

            for step in range(2000):
                params, optimizer_state, loss = train_step(params, T, optimizer_state, precomputed_slices)
                factors_binary = sigmoid(params) > 0.5
                reconstructed_tensor = reconstruct_from_multi_binary_factors(factors_binary)
                residual = get_residual_num(T, reconstructed_tensor)

                if residual < best_residual:
                    best_residual = residual
                    best_rank = rank
                    best_factors = factors_binary
                    if residual == 0:
                        break

                residual_change = prev_residual - residual
                if residual_change < residual_tolerance:
                    plateau_count +=1
                    if plateau_count > max_plateau:
                        break
                else:
                    plateau_count = 0

                prev_residual = residual

        results.append({
            "rank": int(best_rank),
            "residual": int(best_residual),
            "factors": best_factors
        })
    return results
