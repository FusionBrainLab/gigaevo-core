import jax
import jax.numpy as jnp
import numpy as np
import optax

# -----------------------------------------------------------------------------
# JAX Objective and Utility Functions
# -----------------------------------------------------------------------------


@jax.jit
def compute_metrics(f):
    """
    Computes L2^2, L1, and L_inf norms of the autoconvolution g = f * f.
    """
    # Linear convolution of f with itself
    g = jnp.convolve(f, f, mode="full")

    l2_sq = jnp.sum(jnp.square(g))
    l1 = jnp.sum(g)
    l_inf = jnp.max(g)

    return l2_sq, l1, l_inf


@jax.jit
def objective(params):
    """
    Objective function to minimize: -C(f)
    f is parameterized as |params| to ensure non-negativity.
    """
    f = jnp.abs(params)
    l2_sq, l1, l_inf = compute_metrics(f)

    # Calculate C = ||g||_2^2 / (||g||_1 * ||g||_inf)
    # Add small epsilon to denominator to prevent division by zero
    denom = l1 * l_inf + 1e-12
    C = l2_sq / denom

    return -C


# -----------------------------------------------------------------------------
# Improver Class
# -----------------------------------------------------------------------------


class Improver:
    def __init__(self, seed: int = 0):
        """
        Initialize with reproducible random state.
        """
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)

        # Optimization Hyperparameters
        self.target_resolution = 2048
        self.learning_rate = 1e-2
        self.coarse_steps = 1000
        self.fine_steps = 3000

    def _run_optimization(self, current_params, steps):
        """
        Helper method to run Adam optimization loop for a given number of steps.
        """
        optimizer = optax.adam(learning_rate=self.learning_rate)
        opt_state = optimizer.init(current_params)

        @jax.jit
        def update_step(carrier, _):
            params, opt_state = carrier
            loss, grads = jax.value_and_grad(objective)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        # Use jax.lax.scan for efficient execution of the loop
        (final_params, final_state), losses = jax.lax.scan(
            update_step, (current_params, opt_state), None, length=steps
        )
        return final_params

    def improve(self, input_f: np.ndarray) -> np.ndarray:
        """
        Refines an existing function f using rigorous local optimization.
        Uses a Multigrid approach: Optimize Coarse -> Upsample -> Optimize Fine.
        """
        f_jax = jnp.array(input_f)
        current_N = f_jax.shape[0]
        params = f_jax

        # Phase 1: Adaptive Upsampling Strategy
        if current_N < self.target_resolution:
            # Step 1.1: Optimize at the current coarse resolution to set global structure
            # This is faster and helps avoid local optima before adding complexity.
            params = self._run_optimization(params, self.coarse_steps)

            # Step 1.2: Upsample to target resolution
            # We use linear interpolation on the absolute values (the actual function f)
            f_coarse = jnp.abs(params)
            x_old = jnp.linspace(0, 1, current_N)
            x_new = jnp.linspace(0, 1, self.target_resolution)
            f_fine = jnp.interp(x_new, x_old, f_coarse)

            params = f_fine

        # Phase 2: Fine-grained Optimization
        # Run a longer optimization routine on the high-resolution grid
        params = self._run_optimization(params, self.fine_steps)

        # Phase 3: Post-processing
        # Ensure non-negativity and normalize (scale-invariance of C allows normalization)
        final_f = jnp.abs(params)
        final_f = final_f / (jnp.max(final_f) + 1e-12)

        return np.array(final_f)

    def perturb(self, input_f: np.ndarray, intensity: float) -> np.ndarray:
        """
        Applies random noise to the function to help escape local optima.
        """
        self.key, subkey = jax.random.split(self.key)

        # Additive noise
        noise = jax.random.normal(subkey, input_f.shape) * intensity
        perturbed = input_f + noise

        # Ensure non-negative
        return np.array(np.abs(perturbed))

    def generate_config(self, initial_resolution: int = 1000) -> np.ndarray:
        """
        Generates a valid starting function f.
        Starts with random uniform noise, which allows the optimizer to
        carve out the necessary spectral features.
        """
        self.key, subkey = jax.random.split(self.key)

        # Uniform random initialization in [0, 1]
        f = jax.random.uniform(subkey, (initial_resolution,), minval=0.0, maxval=1.0)

        return np.array(f)


def entrypoint():
    return Improver
