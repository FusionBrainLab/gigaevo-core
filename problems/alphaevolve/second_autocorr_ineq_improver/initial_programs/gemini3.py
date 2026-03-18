import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.signal


class Improver:
    """
    Algorithm Explanation:
    The goal is to maximize the second autocorrelation constant $C(f)$ for a non-negative function $f$.
    We define the objective function based on the simplified identity for non-negative functions:
    $$ C(f) = \\frac{\\|f \\star f\\|_2^2}{\\|f\\|_1^2 \\|f\\|_2^2} $$
    where the denominator terms $\|f \\star f\|_1 = \|f\|_1^2$ and $\|f \\star f\|_\\infty = \|f\|_2^2$ hold due to non-negativity
    and the maximum of autocorrelation occurring at lag 0.

    Implementation Strategy:
    1.  **Parameterization**: We model $f(x) = w(x)^2$ to enforce the non-negativity constraint $f(x) \\ge 0$ naturally
        during unconstrained optimization of weights $w$.
    2.  **Optimization**: We use JAX for automatic differentiation and the Adam optimizer (via Optax) to maximize $C(f)$.
        We employ a cosine decay learning rate schedule to settle into optima.
    3.  **Adaptive Resolution**: The solver checks the resolution $N$. If $N < 1024$, it upsamples the function to meet
        the minimum resolution requirement and allow for finer structural adjustments.
    4.  **Perturbation**: To escape local optima, we apply stochastic perturbations including resolution scaling (zooming),
        adding noise, rolling (shifting), and power transformations (sharpening/flattening).
    """

    def __init__(self, seed: int = 0):
        """
        Initialize with reproducible random state.
        """
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        # We prefer float32 for speed on JAX, but float64 is safer for precision if available.
        # JAX defaults to float32.

    def improve(self, input_f: np.ndarray) -> np.ndarray:
        """
        Refines an existing function f using rigorous local optimization.
        Uses JAX for gradient descent on the latent square-root parameters.
        """
        # 1. Resolution Check & Adjustment
        # Ensure minimum resolution of 1024 as per constraints
        target_min = 1024
        f_np = input_f.astype(np.float32)

        if len(f_np) < target_min:
            # Upsample using Fourier method (scipy.signal.resample)
            f_np = scipy.signal.resample(f_np, target_min)
            # Fix potential ringing (negative values) from resampling
            f_np = np.maximum(f_np, 0)

        # 2. Prepare JAX Data
        # Parameterize f = w^2 to enforce non-negativity
        w_init = jnp.array(np.sqrt(f_np + 1e-12))

        # 3. Define Optimization Components locally to close over configuration

        def loss_fn(w):
            """
            Negative of the Objective C(f).
            C = ||f*f||_2^2 / (||f||_1^2 * ||f||_2^2)
            """
            f = w**2

            # Normalize for numerical stability (ratio is scale invariant)
            # Adding epsilon to sum to prevent div by zero
            scale = jnp.sum(f) + 1e-12
            f_norm = f / scale

            # Denominator terms
            # ||f_norm||_1 is 1.0 by definition
            # ||f_norm||_2^2
            norm_sq_f = jnp.sum(f_norm**2)

            # Numerator term: ||f_norm * f_norm||_2^2
            # We use fftconvolve. mode='full' is standard for autocorrelation norms.
            g = jax.scipy.signal.fftconvolve(f_norm, f_norm, mode="full")
            norm_sq_g = jnp.sum(g**2)

            # Calculate C
            # Denom = ||f||_1^2 * ||f||_2^2
            # For normalized f, ||f||_1 = 1.
            denom = norm_sq_f + 1e-12

            C = norm_sq_g / denom

            return -C

        # Configure Optimizer
        steps = 1500
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-4,
            peak_value=0.01,
            warmup_steps=100,
            decay_steps=steps,
            end_value=1e-5,
        )

        optimizer = optax.adam(learning_rate=lr_schedule)
        opt_state = optimizer.init(w_init)

        # Extract pure functions to avoid passing NamedTuple to JIT
        update_fn = optimizer.update

        # Define JIT-compiled update step
        @jax.jit
        def step_fn(w, state):
            loss, grads = jax.value_and_grad(loss_fn)(w)
            updates, new_state = update_fn(grads, state, w)
            new_w = optax.apply_updates(w, updates)
            return new_w, new_state, loss

        # 4. Run Optimization Loop
        # We use lax.scan for efficient execution of the loop in XLA
        def scan_body(carrier, _):
            w, state = carrier
            w_new, state_new, val = step_fn(w, state)
            return (w_new, state_new), val

        (w_final, _), _ = jax.lax.scan(
            scan_body, (w_init, opt_state), None, length=steps
        )

        # 5. Reconstruct Result
        f_final = np.array(w_final**2)

        # Post-processing cleanup
        # Zero out effective zeros
        f_final[f_final < 1e-6 * np.max(f_final)] = 0.0

        # Safety check for all zeros
        if np.sum(f_final) < 1e-12:
            return f_np  # Return original/resampled if optimization failed

        return f_final

    def perturb(self, input_f: np.ndarray, intensity: float) -> np.ndarray:
        """
        Applies discrete, structural, or resolution modifications.
        """
        # Use numpy RNG for perturbations
        rng = np.random.default_rng(self.seed)
        # Update seed for next call
        self.seed += 1

        f = input_f.copy()
        N = len(f)

        # Choose perturbation strategy
        # Strategies:
        # 1. Resize (change resolution)
        # 2. Add Noise (exploration)
        # 3. Roll (shift phase)
        # 4. Power (sharpen/flatten distribution)

        action = rng.choice(
            ["resize", "noise", "roll", "power"], p=[0.3, 0.3, 0.2, 0.2]
        )

        if action == "resize":
            # Adaptive Resolution: Scale between 0.8x and 1.5x depending on intensity
            # Intensity 1e-3 -> scale close to 1. Intensity 1e3 -> wild scaling.
            # We dampen the intensity for scaling to avoid exploding sizes excessively.
            factor_dev = 0.5 * np.tanh(intensity / 10.0)  # bounded deviation
            scale_factor = 1.0 + rng.uniform(-factor_dev, factor_dev)

            new_N = int(N * scale_factor)
            new_N = np.clip(new_N, 1024, 100000)

            if new_N != N:
                f = scipy.signal.resample(f, new_N)
                f = np.maximum(f, 0)

        elif action == "noise":
            # Add sparse noise or dense noise
            sigma = 0.05 * np.max(f) * np.tanh(intensity)
            noise = rng.normal(0, sigma, size=N)
            # Mask noise to be somewhat sparse? Or just add.
            f = f + noise
            f = np.maximum(f, 0)

        elif action == "roll":
            shift_pct = 0.1 * np.tanh(intensity)
            shift = int(rng.normal(0, N * shift_pct))
            f = np.roll(f, shift)

        elif action == "power":
            # Raises elements to a power to sharpen peaks or flatten base
            # p > 1 makes it spikier (good for autocorrelation C usually)
            p = 1.0 + rng.normal(0, 0.2) * np.tanh(intensity)
            p = np.clip(p, 0.5, 3.0)
            f = f**p

        return f

    def generate_config(self, initial_resolution: int = 1000) -> np.ndarray:
        """
        Generates a valid starting function f.
        """
        N = max(initial_resolution, 1024)
        rng = np.random.default_rng(self.seed)

        # A random sparse initialization is often better than uniform or gaussian
        # for autocorrelation maximization (finding optimal spacing/Sidon sets).
        f = np.zeros(N)

        # Populate with random spikes
        density = 10.0 / np.sqrt(N)  # Heuristic for sparse sets
        num_spikes = int(N * density)
        if num_spikes < 2:
            num_spikes = 2

        indices = rng.choice(N, size=num_spikes, replace=False)
        f[indices] = rng.uniform(0.5, 1.5, size=num_spikes)

        # Add very small background noise to prevent zero-gradient issues initially
        f += 1e-4

        return f


def entrypoint():
    return Improver
