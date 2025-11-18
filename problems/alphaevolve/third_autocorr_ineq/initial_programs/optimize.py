import jax
import numpy as np
import optax

from helper import compute_c


def entrypoint() -> np.ndarray:
    num_intervals = 400
    learning_rate = 0.005
    num_steps = 20000
    warmup_steps = 2000

    def objective_fn(f_values):
        return compute_c(f_values)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_steps - warmup_steps,
        end_value=learning_rate * 1e-4,
    )

    optimizer = optax.adam(learning_rate=schedule)

    key = jax.random.PRNGKey(42)
    f_values = jax.random.normal(key, (num_intervals,))

    opt_state = optimizer.init(f_values)

    @jax.jit
    def train_step(f_vals, opt_st):
        loss, grads = jax.value_and_grad(objective_fn)(f_vals)
        updates, opt_st = optimizer.update(grads, opt_st, f_vals)
        f_vals = optax.apply_updates(f_vals, updates)
        return f_vals, opt_st, loss

    for step in range(num_steps):
        f_values, opt_state, loss = train_step(f_values, opt_state)

    return np.array(f_values)
