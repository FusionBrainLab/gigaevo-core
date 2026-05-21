import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

# Set JAX to use 64-bit precision for stable L-BFGS optimization
jax.config.update("jax_enable_x64", True)


def entrypoint(n=600, d=11, seed=0) -> np.ndarray:
    """
    Calculates the optimal spherical code configuration.

    Algorithm Explanation:
    To generate $n$ points on the unit hypersphere $S^{d-1}$ in \mathbb{R}^d that minimize the
    maximum pairwise inner product (the coherence), we approach the problem via an optimization
    framework. The objective is to minimize:
    $$ \mu(X) = \max_{1 \leq i < j \leq n} (x_i \cdot x_j) $$
    subject to $\|x_i\|_2 = 1$ for all $i=1 \dots n$.

    Since the maximum function is non-smooth, we approximate it using the smooth, differentiable
    LogSumExp (LSE) function over all non-diagonal pairs:
    $$ L_\alpha(X) = \frac{1}{\alpha} \log \left( \sum_{i \neq j} \exp(\alpha (x_i \cdot x_j)) \right) $$
    As $\alpha \to \infty$, $L_\alpha(X) \approx \max_{i \neq j} (x_i \cdot x_j)$.

    We define unconstrained variables $Y \in \mathbb{R}^{n \times d}$ and project them onto the
    sphere $X = Y / \|Y\|_2$. To prevent scale drift during optimization, we add a soft penalty
    $\frac{1}{2} \sum (\|y_i\|_2 - 1)^2$.

    The optimization is performed using the L-BFGS-B algorithm, combined with a continuation
    method (annealing schedule) for $\alpha$. We start with a relatively small $\alpha$
    (e.g., 10) to gently spread the points apart and avoid local minima. We then gradually
    increase $\alpha$ up to a large value (e.g., 2000) to fine-tune the configuration and
    strictly minimize the maximum inner product. JAX is used for exact automatic differentiation.

    Args:
        n: Number of points.
        d: Dimension of the space (points lie on S^{d-1} in R^d).
        seed: Random seed for reproducibility.

    Returns:
        points: np.ndarray of shape (n, d) representing the optimized configuration.
    """
    np.random.seed(seed)

    # Random uniform initialization on the sphere
    X = np.random.randn(n, d)
    # Theoretically prevent zero-norm initialization
    norms_init = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.where(norms_init < 1e-6, np.eye(d)[0], X)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    Y0 = X.flatten()

    # Precompute logical mask for diagonal exclusion
    eye_mask = jnp.arange(n) == jnp.arange(n)[:, None]

    @jax.jit
    def loss_and_grad(Y_flat, alpha):
        Y = Y_flat.reshape((n, d))

        # Add epsilon inside sqrt to completely prevent NaN gradients at strictly zero norm
        norms = jnp.sqrt(jnp.sum(Y**2, axis=1, keepdims=True) + 1e-12)
        X_proj = Y / norms

        # Compute pairwise inner products
        ip = jnp.dot(X_proj, X_proj.T)

        # Exclude diagonal elements by replacing them with -infinity
        ip = jnp.where(eye_mask, -jnp.inf, ip)

        # Smooth maximum using LogSumExp
        lse = jax.scipy.special.logsumexp(alpha * ip.flatten()) / alpha

        # Penalty functional to keep Y stably clustered around the unit sphere
        penalty = 0.5 * jnp.sum((norms - 1.0) ** 2)

        return lse + penalty

    # JAX function to compute both objective scalar and analytical gradient vector
    vg = jax.value_and_grad(loss_and_grad)

    def fun(y_np, alpha):
        # SciPy L-BFGS-B demands strict fp64 numpy arrays
        l, g = vg(y_np, alpha)
        return np.array(l, dtype=np.float64), np.array(g, dtype=np.float64)

    Y = Y0
    # Annealing schedule for LSE strictness mapping
    alphas = [10.0, 50.0, 200.0, 500.0, 1000.0, 2000.0]

    for alpha in alphas:
        res = scipy.optimize.minimize(
            fun,
            Y,
            args=(alpha,),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2500, "ftol": 1e-9, "gtol": 1e-5},
        )
        Y = res.x

    # Retrieve the final optimized representation map
    Y_mat = Y.reshape((n, d))

    # Final exact geometric projection strictly enforcing constraint norm = 1.0
    final_norms = np.linalg.norm(Y_mat, axis=1, keepdims=True)
    final_norms[final_norms < 1e-12] = 1.0  # Fallback for unlikely collapsed norm
    X_final = Y_mat / final_norms

    return X_final
