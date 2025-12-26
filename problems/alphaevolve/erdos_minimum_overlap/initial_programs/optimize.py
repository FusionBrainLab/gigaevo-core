import jax
import jax.numpy as jnp
import optax
import numpy as np

from helper import compute_c


def entrypoint() -> np.ndarray:
    num_intervals = 200
    learning_rate = 0.005
    num_steps = 20000
    penalty_strength = 1000000.0
    domain_width = 2.0
    dx = domain_width / num_intervals

    def objective_fn(latent_h_values: jnp.ndarray) -> jnp.ndarray:
        h = jax.nn.sigmoid(latent_h_values)
        objective_loss = compute_c(h)
        
        integral_h = jnp.sum(h) * dx
        constraint_loss = (integral_h - 1.0) ** 2
        
        total_loss = objective_loss + penalty_strength * constraint_loss
        return total_loss

    optimizer = optax.adam(learning_rate)

    key = jax.random.PRNGKey(42)
    latent_h_values = jax.random.normal(key, (num_intervals,))

    opt_state = optimizer.init(latent_h_values)

    @jax.jit
    def train_step(latent_h_values, opt_state):
        loss, grads = jax.value_and_grad(objective_fn)(latent_h_values)
        updates, opt_state = optimizer.update(grads, opt_state)
        latent_h_values = optax.apply_updates(latent_h_values, updates)
        return latent_h_values, opt_state, loss

    for step in range(num_steps):
        latent_h_values, opt_state, loss = train_step(latent_h_values, opt_state)

    final_h = jax.nn.sigmoid(latent_h_values)

    return np.array(final_h)

