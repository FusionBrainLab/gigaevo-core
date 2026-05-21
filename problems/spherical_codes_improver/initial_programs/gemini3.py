"""
================================================================================
D-DIMENSIONAL SPHERICAL CODE OPTIMIZER
================================================================================

This module implements a state-of-the-art optimizer for discovering spherical
codes, which aims to distribute $n$ points on the unit sphere $S^{d-1}$ in
$\mathbb{R}^d$ such that the maximum pairwise inner product (cosine similarity)
is minimized. This is equivalent to solving the Tammes problem (maximizing the
minimum angle).

### 1. Improvement Methodology (Surrogate Optimization)
Directly minimizing the maximum inner product is a non-smooth minimax problem:
    \min_{X} \max_{i < j} (x_i \cdot x_j)
To optimize this efficiently using gradient-based methods, we employ a smooth
surrogate based on the LogSumExp (LSE) function:
    L(X, \alpha) = \frac{1}{\alpha} \log \sum_{i \neq j} \exp(\alpha (x_i \cdot x_j))
where $\alpha > 0$ is a temperature parameter. As $\alpha \to \infty$, the LSE
smoothly approximates the true maximum. In the `improve` method, we apply the
Adam optimizer via `optax` in a highly-optimized JAX scan loop. We dynamically scale
$\alpha$ from a small value (for global energy smoothing) to a large value (to
strictly minimize the local bottlenecks) over the optimization steps, along with
an exponentially decaying learning rate.

### 2. Perturbation Methodology (Active-Set Targeting & Landscape Traversal)
High-quality spherical codes reside in a rugged, non-convex energy landscape
dominated by symmetries and localized bottlenecks. Applying global random noise
destroys correctly packed subsets. Instead, our `perturb` method functions as an
orchestrator for targeted topological shifts:
    - **Active-Set Extraction:** We isolate the top $k$ points participating in
      the tightest pairwise collisions. $k$ is modulated by the `intensity` parameter.
    - **Localized Vector Fields:** For each bottleneck point, we identify its
      nearest neighbor and compute a repulsive direction.
    - **Tangent-Space Momentum:** We blend the repulsive vector with Gaussian
      noise (scaled by intensity) and project the resultant vector onto the
      tangent space of $S^{d-1}$ at the point.
    - **Geodesic Traversal:** The active points are then moved along the geodesics
      of the hypersphere by a structural angle determined by the `intensity`. This
      breaks local symmetries without disturbing the well-packed global majority.
================================================================================
"""

import jax
import numpy as np

# Enable double precision for sensitive geometric constraints
jax.config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp
import optax


# Define JAX-JITted functions at the module level to avoid recompilation issues.
@jax.jit
def get_loss(Y: jnp.ndarray, alpha: float) -> jnp.ndarray:
    # Normalization constraint projection
    norms = jnp.linalg.norm(Y, axis=1, keepdims=True)
    X = Y / jnp.clip(norms, 1e-12, None)

    # Pairwise cosine similarities
    dot_products = jnp.dot(X, X.T)

    # Mask out self-interactions (diagonal)
    mask = jnp.eye(Y.shape[0], dtype=bool)
    masked_dots = jnp.where(mask, -jnp.inf, dot_products)

    # Smooth LogSumExp surrogate for the maximum pairwise dot product
    max_dot = jax.scipy.special.logsumexp(masked_dots * alpha) / alpha
    return max_dot


# Steps, alpha schedules, and learning rates are declared as static arguments.
@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def optimize_loop(
    Y_init: jnp.ndarray,
    steps: int,
    alpha_start: float,
    alpha_end: float,
    lr_start: float,
    lr_end: float,
) -> jnp.ndarray:
    # Dynamic schedule to smoothly morph the landscape from global to precise
    alpha_schedule = jnp.linspace(alpha_start, alpha_end, steps)

    # Exponentially decay learning rate to converge tightly at the bottom of the basin
    decay_rate = lr_end / lr_start
    lr_schedule = optax.exponential_decay(
        init_value=lr_start, transition_steps=steps, decay_rate=decay_rate
    )

    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(Y_init)

    def step_fn(carry, i):
        Y, state = carry
        alpha = alpha_schedule[i]
        loss, grads = jax.value_and_grad(get_loss)(Y, alpha)
        updates, state = optimizer.update(grads, state, Y)

        # Apply updates directly rather than via optax.apply_updates
        # to satisfy strict pyright analysis when dealing with single arrays.
        Y_next = Y + updates

        # Explicit reprojection onto the sphere at each step to prevent norm drift
        norms = jnp.linalg.norm(Y_next, axis=1, keepdims=True)
        Y_next = Y_next / jnp.clip(norms, 1e-12, None)

        return (Y_next, state), loss

    # Unroll optimization into a fast C++/XLA scan loop
    (Y_final, _), losses = jax.lax.scan(step_fn, (Y_init, opt_state), jnp.arange(steps))
    return Y_final


class Improver:
    def __init__(self, n: int, d: int, seed: int = 0):
        """
        Args:
            n: Number of points.
            d: Dimension of the space (points lie on S^{d-1} in R^d).
            seed: Random seed.
        """
        self.n = n
        self.d = d
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_config(self, seed=None) -> np.ndarray:
        """
        Generates a valid initial configuration of n points uniformly distributed on the unit hypersphere.
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        points = rng.standard_normal((self.n, self.d))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms

    def improve(self, points: np.ndarray, seed=None) -> np.ndarray:
        """
        Refines an existing configuration to minimize the maximum inner product.
        Strictly enforces the spherical constraint (||x|| = 1) in the output.
        """
        # Hyperparameters for the internal surrogate optimization
        steps = 2000
        alpha_start = 10.0
        alpha_end = 300.0
        lr_start = 0.05
        lr_end = 0.0005

        # Load constraints onto the compiled JAX landscape
        Y_init = jnp.array(points)
        Y_opt = optimize_loop(Y_init, steps, alpha_start, alpha_end, lr_start, lr_end)

        # Project back to sphere exactly utilizing robust numpy (float64 precision)
        Y_opt_np = np.array(Y_opt, dtype=np.float64)
        norms = np.linalg.norm(Y_opt_np, axis=1, keepdims=True)
        return Y_opt_np / norms

    def perturb(self, points: np.ndarray, intensity: float, seed=None) -> np.ndarray:
        """
        Applies active-set active targeting and tangent space projections to intelligently
        escape local minima while rigorously preserving the carefully packed substructures.
        """
        rng = np.random.default_rng(
            seed if seed is not None else self.rng.integers(0, 2**31)
        )

        # Identify bottlenecks (points sharing the highest collision cosines)
        dots = points @ points.T
        np.fill_diagonal(dots, -np.inf)
        max_dots_per_point = np.max(dots, axis=1)

        worst_indices = np.argsort(-max_dots_per_point)

        # Scale active-set size sub-linearly based on the given intensity
        k = int(np.clip(self.n * (intensity**1.5), 2, self.n))
        active_set = worst_indices[:k]

        new_points = points.copy()

        for idx in active_set:
            jdx = int(np.argmax(dots[idx]))
            repel_dir = points[idx] - points[jdx]

            norm_repel = float(np.linalg.norm(repel_dir))
            if norm_repel > 1e-8:
                repel_dir /= norm_repel
            else:
                repel_dir = rng.standard_normal(self.d)
                repel_dir /= float(np.linalg.norm(repel_dir))

            noise = rng.standard_normal(self.d)
            noise_norm = float(np.linalg.norm(noise))
            if noise_norm > 1e-8:
                noise /= noise_norm

            # Morph between structured repulsive fields and stochastic basin-exploding momentum
            direction = (1.0 - intensity) * repel_dir + intensity * noise

            # Target topological limits by confining traversal to the specific point's tangent space
            tangent_dir = (
                direction - float(np.dot(direction, points[idx])) * points[idx]
            )

            norm_tangent = float(np.linalg.norm(tangent_dir))
            if norm_tangent > 1e-8:
                tangent_dir /= norm_tangent
                # Base rotation dynamically responds to the depth of the local trap (encoded in intensity)
                base_angle = (intensity**0.5) * (np.pi / 3.0)
                angle = base_angle * rng.uniform(0.5, 1.0)

                # Execute Exact geodesic traversal
                new_points[idx] = (
                    np.cos(angle) * points[idx] + np.sin(angle) * tangent_dir
                )

        # Micro-correction against floating point drift
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / np.clip(norms, 1e-12, None)

        return new_points


def entrypoint():
    """
    Returns the target class.
    """
    return Improver
