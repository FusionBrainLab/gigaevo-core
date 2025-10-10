"""
Top Program #21
Program ID: 4fec2074-9398-4a08-9c8a-19b8d04e9dae
Fitness: 8.0381
Created: 2025-10-09 17:47:36.391159+00:00
Updated: 2025-10-09 19:11:35.264307+00:00
Generation: 2
State: ProgramState.EVOLVING
"""

# Insight (algebraic): Utilizing a greedy approach to iteratively find factors that reduce the residual tensor.
# Insight (optimization): Employing a multi-start strategy with different initializations for each factor search to escape local minima.
# Lineage (generalization): Generalizing the idea of iterative factor finding by incorporating multiple restarts and a more robust factor search.
# Design: Iterative greedy search for factors, with a multi-start optimization for each factor, and a final refinement step.
from helper import Data, get_residual_num, reconstruct_from_multi_binary_factors, reconstruct_from_single_binary_factor
import jax
import jax.numpy as jnp
import optax

def entrypoint(context: list[Data]) -> list[dict]:
    results = []
    for data in context:
        tensor = data.tensor
        n = tensor.shape[0]
        sota_rank = data.sota_rank

        best_rank = float('inf')
        best_factors = None
        best_residual = float('inf')

        # Try ranks from 1 up to sota_rank + a small displacement
        max_rank_to_test = sota_rank + 5

        for current_rank_limit in range(1, max_rank_to_test + 1):
            current_factors = []
            current_residual_tensor = tensor
            
            # Greedy search for factors
            for _ in range(current_rank_limit):
                best_local_factor = None
                min_local_residual = float('inf')

                # Multi-start for each factor search
                num_restarts = 5 
                for _ in range(num_restarts):
                    # Initialize factor with random values
                    key = jax.random.PRNGKey(jnp.sum(jnp.array(current_factors, dtype=jnp.uint8)) + jax.random.randint(jax.random.PRNGKey(0), (), 0, 1000000))
                    initial_factor_float = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0, dtype=jnp.float32)

                    # Optimize a single factor
                    factor_params = {'factor': initial_factor_float}
                    optimizer = optax.adam(learning_rate=0.01)
                    opt_state = optimizer.init(factor_params)

                    @jax.jit
                    def loss_fn(params, target_tensor, current_known_factors_binary):
                        current_factor_binary = (jax.nn.sigmoid(params['factor']) > 0.5).astype(jnp.uint8)
                        
                        # Reconstruct with current factor and known factors
                        if current_known_factors_binary.size > 0:
                            all_factors_binary = jnp.concatenate([current_known_factors_binary, current_factor_binary[:, None]], axis=1)
                            reconstructed_tensor = reconstruct_from_multi_binary_factors(all_factors_binary)
                        else:
                            reconstructed_tensor = reconstruct_from_single_binary_factor(current_factor_binary)
                        
                        # Use float32 for loss calculation to enable gradients
                        diff = (target_tensor.astype(jnp.float32) - reconstructed_tensor.astype(jnp.float32))
                        return jnp.sum(diff * diff)

                    @jax.jit
                    def update_step(params, opt_state, target_tensor, current_known_factors_binary):
                        loss, grads = jax.value_and_grad(loss_fn)(params, target_tensor, current_known_factors_binary)
                        updates, opt_state = optimizer.update(grads, opt_state, params)
                        params = optax.apply_updates(params, updates)
                        return params, opt_state, loss

                    # Convert current_factors to uint8 for loss_fn
                    current_factors_binary_for_loss = jnp.array(current_factors, dtype=jnp.uint8).T if current_factors else jnp.array([], dtype=jnp.uint8)

                    for _ in range(200): # Optimization steps for a single factor
                        factor_params, opt_state, loss = update_step(factor_params, opt_state, current_residual_tensor, current_factors_binary_for_loss)

                    optimized_factor_binary = (jax.nn.sigmoid(factor_params['factor']) > 0.5).astype(jnp.uint8)
                    
                    # Calculate residual with the new factor
                    temp_factors = current_factors + [optimized_factor_binary]
                    temp_reconstructed = reconstruct_from_multi_binary_factors(jnp.array(temp_factors, dtype=jnp.uint8).T)
                    temp_residual_num = get_residual_num(tensor, temp_reconstructed)

                    if temp_residual_num < min_local_residual:
                        min_local_residual = temp_residual_num
                        best_local_factor = optimized_factor_binary

                if best_local_factor is not None:
                    current_factors.append(best_local_factor)
                else:
                    break # Could not find a suitable factor

                # Update the residual tensor for the next iteration
                reconstructed_so_far = reconstruct_from_multi_binary_factors(jnp.array(current_factors, dtype=jnp.uint8).T)
                current_residual_tensor = tensor ^ reconstructed_so_far

            if current_factors:
                final_reconstructed_tensor = reconstruct_from_multi_binary_factors(jnp.array(current_factors, dtype=jnp.uint8).T)
                final_residual_num = get_residual_num(tensor, final_reconstructed_tensor)
                
                if final_residual_num == 0 and len(current_factors) < best_rank:
                    best_rank = len(current_factors)
                    best_factors = jnp.array(current_factors, dtype=jnp.uint8).T
                    best_residual = 0
                    break # Found an exact decomposition, no need to try higher ranks

        # If no exact decomposition found, return the best found so far with minimal residual
        if best_factors is None and current_factors:
             final_reconstructed_tensor = reconstruct_from_multi_binary_factors(jnp.array(current_factors, dtype=jnp.uint8).T)
             final_residual_num = get_residual_num(tensor, final_reconstructed_tensor)
             best_rank = len(current_factors)
             best_factors = jnp.array(current_factors, dtype=jnp.uint8).T
             best_residual = final_residual_num
        elif best_factors is None: # Fallback if nothing was found
             best_rank = sota_rank + 1
             best_factors = jnp.zeros((n, best_rank), dtype=jnp.uint8)
             best_residual = get_residual_num(tensor)


        results.append({
            "rank": best_rank,
            "residual": best_residual,
            "factors": best_factors
        })
    return results
