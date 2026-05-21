import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize


def entrypoint(n=600, d=11, seed=0) -> np.ndarray:
    """
    Calculates the optimal spherical code configuration.

    Algorithm Explanation:
    This algorithm solves the Grassmannian packing / spherical code problem by minimizing
    the maximum pairwise cosine similarity between $n$ points on the unit sphere $S^{d-1}$.
    The true objective function is the minimax problem: $\mu(X) = \max_{i < j} (x_i \cdot x_j)$.
    Since the pure min-max objective is non-smooth, we use the smooth, differentiable
    LogSumExp approximation:
    $$ L_\alpha(X) = \frac{1}{\alpha} \log \sum_{i < j} \exp(\alpha (x_i \cdot x_j)) $$
    As $\alpha \to \infty$, $L_\alpha(X)$ converges strictly to $\mu(X)$.
    We optimize this objective using the L-BFGS-B algorithm. To avoid getting trapped in
    suboptimal local minima and to handle the non-smoothness gracefully, we employ a
    continuation method (staged optimization), solving the problem for an increasing
    sequence of $\alpha$ values (from 10 to 4000). Points are parameterized in $\mathbb{R}^d$
    and projected onto the unit sphere, with a small $L_2$ penalty to maintain their
    pre-projection norm near 1.

    Args:
        n: Number of points.
        d: Dimension of the space (points lie on S^{d-1} in R^d).
        seed: Random seed for reproducibility.

    Returns:
        points: np.ndarray of shape (n, d) representing the optimized configuration.
    """
    np.random.seed(seed)

    # We define the objective inside so it captures dynamic `n` and `d` properly
    # JAX will compile a specific version for the given `n` and `d` sizes.
    @jax.jit
    def objective(Y_flat, alpha):
        Y = Y_flat.reshape((n, d))
        # Determine the lengths for projecting to the unit sphere
        norms = jnp.linalg.norm(Y, axis=1, keepdims=True)
        # Add a tiny epsilon to prevent any risk of division by zero
        X = Y / (norms + 1e-12)

        # Matrix of all pairwise dot products (cosine similarities)
        Z = jnp.dot(X, X.T)

        # Upper triangle mask to efficiently pick unique pairs i < j
        mask = jnp.triu(jnp.ones((n, n)), k=1)
        # Ignore self-interactions (diagonal) and duplicate pairs by masking with -1000
        # (Since dot product is in [-1, 1], -1000 is safely treated as -infinity)
        Z_masked = jnp.where(mask == 1, Z, -1000.0)

        max_Z = jnp.max(Z_masked)

        # LogSumExp (numerically stable implementation)
        lse = (
            max_Z + jnp.log(jnp.sum(jnp.exp(alpha * (Z_masked - max_Z)) * mask)) / alpha
        )

        # Weak regularization to prevent scale drifting, keeping the base points near norm 1
        reg = 1e-4 * jnp.sum((norms - 1.0) ** 2)

        return lse + reg

    # Combine value and gradient calculation in one compiled pass
    value_and_grad_fn = jax.jit(jax.value_and_grad(objective))

    def scipy_obj(Y_flat, alpha):
        val, grad = value_and_grad_fn(Y_flat, alpha)
        # scipy.optimize expects float64 numpy arrays
        return np.array(val, dtype=np.float64), np.array(grad, dtype=np.float64)

    # Initial points: randomly sampled, then projected to the uniform unit sphere
    Y_init = np.random.randn(n, d)
    Y_init /= np.linalg.norm(Y_init, axis=1, keepdims=True)
    Y_flat = np.array(Y_init.flatten(), dtype=np.float64)

    # Continuation method: gradually increase alpha to refine the minimax approximation
    # Starting with lower alpha allows points to globally organize before hardening constraints
    alphas = [10.0, 50.0, 200.0, 1000.0, 4000.0]
    for alpha in alphas:
        res = minimize(
            scipy_obj,
            Y_flat,
            args=(alpha,),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2000, "ftol": 1e-8, "gtol": 1e-5},
        )
        Y_flat = res.x

    # Extract coordinates and explicitly clamp to the surface of the sphere
    Y_final = Y_flat.reshape((n, d))
    X_final = Y_final / np.linalg.norm(Y_final, axis=1, keepdims=True)

    return np.array(X_final, dtype=np.float64)
