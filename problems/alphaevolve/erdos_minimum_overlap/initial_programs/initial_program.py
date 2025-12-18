import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass
import tqdm


@dataclass
class Hyperparameters:
    num_intervals: int = 200
    learning_rate: float = 0.005
    num_steps: int = 20000
    penalty_strength: float = 1000000.0


class ErdosOptimizer:
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 2.0
        self.dx = self.domain_width / self.hypers.num_intervals

    def _objective_fn(self, latent_h_values: jnp.ndarray) -> jnp.ndarray:
        h = jax.nn.sigmoid(latent_h_values)

        j = 1.0 - h
        N = self.hypers.num_intervals
        h_padded = jnp.pad(h, (0, N))
        j_padded = jnp.pad(j, (0, N))
        corr_fft = jnp.fft.fft(h_padded) * jnp.conj(jnp.fft.fft(j_padded))
        correlation = jnp.fft.ifft(corr_fft).real
        scaled_correlation = correlation * self.dx
        objective_loss = jnp.max(scaled_correlation)

        integral_h = jnp.sum(h) * self.dx
        constraint_loss = (integral_h - 1.0) ** 2

        total_loss = objective_loss + self.hypers.penalty_strength * constraint_loss
        return total_loss

    def run_optimization(self):
        optimizer = optax.adam(self.hypers.learning_rate)

        key = jax.random.PRNGKey(42)
        latent_h_values = jax.random.normal(key, (self.hypers.num_intervals,))

        opt_state = optimizer.init(latent_h_values)

        @jax.jit
        def train_step(latent_h_values, opt_state):
            loss, grads = jax.value_and_grad(self._objective_fn)(latent_h_values)
            updates, opt_state = optimizer.update(grads, opt_state)
            latent_h_values = optax.apply_updates(latent_h_values, updates)
            return latent_h_values, opt_state, loss

        for step in range(self.hypers.num_steps):
            latent_h_values, opt_state, loss = train_step(latent_h_values, opt_state)

        final_h = jax.nn.sigmoid(latent_h_values)

        return np.array(final_h)


def entrypoint() -> np.ndarray:
    hypers = Hyperparameters()
    optimizer = ErdosOptimizer(hypers)
    final_h_values = optimizer.run_optimization()
    return final_h_values

