"""Warm-start seed 4 for the ACI ImprovEvolve strong run (NOT the matched 9).

Source: ImprovEvolve/initial_programs/second_autocorr/improver/gemini4.py
(a pre-existing strong ImprovEvolve initial program), embedded verbatim as
_NativeImprover. The Improver adapter below exposes the aci_improve interface
and pins every returned array to exactly the constructor's n on the fixed
N=50000 grid; the native code's own resolution choices are clamped away, so no
output can crank resolution. C is scale-invariant, so no renormalization."""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.signal
import optax
from functools import partial

class _NativeImprover:
    def __init__(self, seed: int = 0):
        """
        Initialize with reproducible random state and optimization settings.
        """
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        # Adam optimizer with a learning rate tuned for this functional landscape
        self.optimizer = optax.adam(learning_rate=1e-2)

    @partial(jax.jit, static_argnums=(0,))
    def _loss_fn(self, params):
        """
        Computes the negative objective ratio for minimization.
        C(f) = ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf)
        """
        # Enforce non-negativity: f(x) >= 0
        f = jnp.abs(params)
        # Add epsilon to prevent all-zeros collapse (though unlikely with initialization)
        f = f + 1e-12
        
        # Discrete Convolution: g = f * f
        # mode='full' results in size 2N - 1
        g = jax.scipy.signal.convolve(f, f, mode='full')
        
        # Compute Norms
        # L2 squared: sum(g^2)
        g_sq_sum = jnp.sum(g**2)
        # L1: sum(g)
        g_sum = jnp.sum(g)
        # L_inf: max(g)
        g_max = jnp.max(g)
        
        # Calculate Ratio
        # Add epsilon to denominator for numerical stability
        ratio = g_sq_sum / (g_sum * g_max + 1e-12)
        
        # Return negative ratio for minimization
        return -ratio

    @partial(jax.jit, static_argnums=(0,))
    def _update_step(self, params, opt_state):
        """
        Performs a single optimization step using Adam.
        """
        grads = jax.grad(self._loss_fn)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def improve(self, input_f: np.ndarray) -> np.ndarray:
        """
        Refines an existing function f using rigorous local optimization.
        
        ADAPTIVE RESOLUTION:
        Upsamples the array to N=1024 if it is smaller, to satisfy constraints 
        and improve precision.
        """
        # 1. Adaptive Resolution Check
        current_N = len(input_f)
        target_N = max(1024, current_N)
        
        if current_N < target_N:
            # Linear interpolation to upsample
            x_old = np.linspace(0, 1, current_N)
            x_new = np.linspace(0, 1, target_N)
            # Use numpy for interpolation before moving to JAX
            input_f = np.interp(x_new, x_old, input_f)
            
        # 2. Initialization
        params = jnp.array(input_f)
        # Normalize to range [0, 1] approximately to keep gradients stable
        # The objective is scale invariant, but optimizer is not.
        max_val = jnp.max(jnp.abs(params))
        params = params / (max_val + 1e-9)
        
        opt_state = self.optimizer.init(params)
        
        # 3. Optimization Loop
        # We run a fixed number of steps. For a continuous improve loop, 
        # the user can call this method repeatedly.
        n_steps = 500
        
        def step(carry, _):
            p, state = carry
            p, state = self._update_step(p, state)
            return (p, state), None
        
        # lax.scan is faster than a python loop due to reduced overhead
        (params, opt_state), _ = jax.lax.scan(step, (params, opt_state), None, length=n_steps)
        
        # 4. Final Formatting
        improved_f = np.abs(np.array(params))
        return improved_f

    def perturb(self, input_f: np.ndarray, intensity: float) -> np.ndarray:
        """
        Applies structural modifications to escape local optima.
        Adds noise proportional to the signal strength.
        """
        self.key, subkey = jax.random.split(self.key)
        
        # Calculate scale based on signal mean to keep perturbation relative
        scale = np.mean(input_f) * intensity
        
        # Generate Gaussian noise
        noise = jax.random.normal(subkey, shape=input_f.shape) * scale
        
        perturbed_f = input_f + np.array(noise)
        
        # Ensure non-negativity is preserved after perturbation
        return np.abs(perturbed_f)

    def generate_config(self, initial_resolution: int = 1000) -> np.ndarray:
        """
        Generates a valid starting function f.
        Random initialization is often superior for finding complex basins 
        in autocorrelation landscapes.
        """
        self.key, subkey = jax.random.split(self.key)
        
        # Uniform random distribution [0, 1]
        # This provides a rich set of frequencies for the optimizer to carve out.
        f = jax.random.uniform(subkey, (initial_resolution,))
        
        return np.array(f)


# --- aci_improve interface adapter (added; the class above is verbatim) -------------
# The native algorithm chose its own resolution; this study fixes n at the
# constructor and grades the returned array AS GIVEN. The adapter routes the three
# aci_improve methods to the native class and clamps every output to exactly length
# self.n, non-negative and finite -- enforcing the fixed grid, never inflating C by
# changing it (which the grader rejects as wrong-length anyway).
import numpy as _np


class Improver:
    def __init__(self, n: int = 50000, seed: int = 0):
        self.n = int(n)
        self.seed = int(seed)
        self._inner = _NativeImprover(seed)

    def _fit(self, f) -> _np.ndarray:
        a = _np.asarray(f, dtype=_np.float64).ravel()
        if a.size == 0:
            return _np.ones(self.n, dtype=_np.float64)
        if a.size != self.n:
            x_old = _np.linspace(0.0, 1.0, a.size)
            x_new = _np.linspace(0.0, 1.0, self.n)
            a = _np.interp(x_new, x_old, a)
        a = _np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = _np.maximum(a, 0.0)
        if not _np.any(a > 0.0):
            return _np.ones(self.n, dtype=_np.float64)
        return a

    def generate_config(self, seed=None) -> _np.ndarray:
        return self._fit(self._inner.generate_config(self.n))

    def improve(self, input_config, seed=None) -> _np.ndarray:
        base = self._fit(input_config)
        return self._fit(self._inner.improve(base))

    def perturb(self, input_config, intensity: float, seed=None) -> _np.ndarray:
        base = self._fit(input_config)
        return self._fit(self._inner.perturb(base, intensity))


def entrypoint():
    return Improver
