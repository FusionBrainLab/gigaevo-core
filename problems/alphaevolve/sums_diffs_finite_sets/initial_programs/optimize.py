import jax
import jax.numpy as jnp
import numpy as np

from helper import compute_c6


def entrypoint() -> np.ndarray:
    max_integer = 250
    num_restarts = 2
    num_search_steps = 300
    initial_temperature = 0.01

    def objective_fn(u_mask):
        U = jnp.where(u_mask)[0]
        c6_value = compute_c6(U)
        return -c6_value

    def anneal_step(key, temp, current_mask, current_loss):
        idx_to_flip = jax.random.randint(key, (), 1, len(current_mask))
        neighbor_mask = current_mask.at[idx_to_flip].set(1 - current_mask[idx_to_flip])

        neighbor_loss = objective_fn(neighbor_mask)
        delta_loss = neighbor_loss - current_loss

        should_accept = False
        if delta_loss < 0:
            should_accept = True
        else:
            accept_prob = jnp.exp(-delta_loss / temp)
            if jax.random.uniform(key) < accept_prob:
                should_accept = True

        if should_accept:
            return neighbor_mask, neighbor_loss
        else:
            return current_mask, current_loss

    main_key = jax.random.PRNGKey(42)

    best_loss = float("inf")
    best_set_np = None

    for i in range(num_restarts):
        restart_key, main_key = jax.random.split(main_key)

        init_key, restart_key = jax.random.split(restart_key)
        sparsity = 0.95
        u_mask = jax.random.bernoulli(init_key, p=(1 - sparsity), shape=(max_integer + 1,))
        u_mask = u_mask.at[0].set(True)

        current_loss = objective_fn(u_mask)

        current_mask = u_mask
        for step in range(num_search_steps):
            restart_key, step_key = jax.random.split(restart_key)
            current_temp = initial_temperature * (1 - step / num_search_steps)
            current_mask, current_loss = anneal_step(
                step_key, jnp.maximum(current_temp, 1e-6), current_mask, current_loss
            )

        final_set = np.where(current_mask)[0]

        if current_loss < best_loss:
            best_loss = current_loss
            best_set_np = final_set

    return best_set_np
