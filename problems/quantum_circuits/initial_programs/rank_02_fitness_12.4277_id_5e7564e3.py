"""
Top Program #2
Program ID: 5e7564e3-704b-4443-b768-dbd84b251ae1
Fitness: 12.4277
Created: 2025-10-09 20:22:18.600091+00:00
Updated: 2025-10-09 20:31:36.700180+00:00
Generation: 2
State: ProgramState.DISCARDED
"""

# Insight (algebraic): Greedy approach can find exact decompositions efficiently by iteratively finding the best single factor.
# Insight (optimization): Using multiple random initializations for factors helps escape local minima.
# Lineage (imitation): Imitates the greedy strategy of iteratively finding factors.
# Design: Implement a greedy search for factors, using an optimized single-factor search at each step with multiple initializations.
import jax
import jax.numpy as jnp
import optax
from helper import Data, get_residual_num, reconstruct_from_multi_binary_factors, reconstruct_from_single_binary_factor


def entrypoint(context: list[Data]) -> list[dict]:
    results = []
    for data in context:
        tensor = data.tensor
        n = tensor.shape[0]
        sota_rank = data.sota_rank

        best_overall_rank = float('inf')
        best_overall_decomposition = None

        # Try a few different greedy runs
        for _ in range(3):  # Number of greedy restarts
            current_residual_tensor = tensor
            current_factors = []
            current_rank = 0

            for r_idx in range(sota_rank + 5):  # Max rank to search for
                if get_residual_num(current_residual_tensor) == 0:
                    break

                best_factor_for_step = None
                min_residual_after_step = float('inf')

                # Multiple initializations for finding the best single factor
                for _init in range(5):
                    key = jax.random.PRNGKey(_init)
                    initial_factor_logits = jax.random.normal(key, (n,), dtype=jnp.float32) * 0.1

                    # Define the loss function for a single factor
                    def single_factor_loss(logits, target_tensor):
                        factor_binary = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.uint8)
                        reconstructed_tensor = reconstruct_from_single_binary_factor(factor_binary)
                        residual_tensor = reconstructed_tensor ^ target_tensor
                        return jnp.sum(residual_tensor).astype(jnp.float32)

                    # Initialize optimizer
                    optimizer = optax.adam(learning_rate=0.1)
                    opt_state = optimizer.init(initial_factor_logits)

                    # Optimization loop for a single factor
                    factor_logits = initial_factor_logits
                    for _iter in range(100):
                        loss_value, grads = jax.value_and_grad(single_factor_loss)(factor_logits, current_residual_tensor)
                        updates, opt_state = optimizer.update(grads, opt_state, factor_logits)
                        factor_logits = optax.apply_updates(factor_logits, updates)

                    optimized_factor_binary = (jax.nn.sigmoid(factor_logits) > 0.5).astype(jnp.uint8)
                    reconstructed_by_factor = reconstruct_from_single_binary_factor(optimized_factor_binary)
                    temp_residual_tensor = current_residual_tensor ^ reconstructed_by_factor
                    temp_residual_num = get_residual_num(temp_residual_tensor)

                    if temp_residual_num < min_residual_after_step:
                        min_residual_after_step = temp_residual_num
                        best_factor_for_step = optimized_factor_binary

                if best_factor_for_step is not None:
                    current_factors.append(best_factor_for_step)
                    current_rank += 1
                    current_residual_tensor = current_residual_tensor ^ reconstruct_from_single_binary_factor(best_factor_for_step)
                else:
                    break # Could not find a factor to improve the residual

            if get_residual_num(current_residual_tensor) == 0 and current_rank < best_overall_rank:
                best_overall_rank = current_rank
                best_overall_decomposition = jnp.stack(current_factors, axis=1) if current_factors else jnp.empty((n, 0), dtype=jnp.uint8)

        if best_overall_decomposition is not None:
            results.append({
                "rank": best_overall_rank,
                "residual": 0,
                "factors": best_overall_decomposition
            })
        else:
            # Fallback if no exact decomposition found (should not happen with sufficient rank search)
            # This part needs to be handled carefully, maybe return the best approximate found.
            # For now, return an empty decomposition if no exact one is found.
            results.append({
                "rank": 0,
                "residual": get_residual_num(tensor),
                "factors": jnp.empty((n, 0), dtype=jnp.uint8)
            })

    return results
