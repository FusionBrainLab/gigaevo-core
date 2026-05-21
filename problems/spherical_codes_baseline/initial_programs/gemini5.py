import jax
import numpy as np

# Ensure float64 precision for numerical stability during optimization
jax.config.update("jax_enable_x64", True)
from jax import jit, value_and_grad
import jax.numpy as jnp
from scipy.optimize import minimize


def entrypoint(n=600, d=11, seed=0) -> np.ndarray:
    """
    Calculates the optimal spherical code configuration.

    Algorithm Explanation:
    The problem of finding a spherical code that minimizes the maximum pairwise inner product
    is equivalent to the Tammes problem (or maximizing the minimum distance on a sphere).
    We solve this by minimizing a smooth approximation of the maximum inner product.

    We employ the log-sum-exp (LSE) function to smooth the min-max objective:
    $$ L(X) = \frac{1}{\alpha} \log \sum_{i < j} \exp(\alpha \langle x_i, x_j \rangle) $$
    where $X \in \mathbb{R}^{n \times d}$ and $\|x_i\| = 1$.

    By gradually increasing the smoothing parameter $\alpha$ (a continuation method), we
    transition from minimizing a global Gaussian-like potential (which prevents local minima
    by uniformly spreading points) to focusing strictly on the closest pairs of points.
    The vectors are parameterized freely in $\mathbb{R}^d$, with L2 normalization applied
    within the forward pass and a soft penalty enforcing $\|x_i\| \approx 1$ to maintain
    well-conditioned gradients. The optimization is carried out using the L-BFGS-B
    algorithm with JAX providing exact and extremely fast gradients.

    Args:
        n: Number of points.
        d: Dimension of the space (points lie on S^{d-1} in R^d).
        seed: Random seed for reproducibility.

    Returns:
        points: np.ndarray of shape (n, d) representing the optimized configuration.
    """
    np.random.seed(seed)

    # Initialize random points on the unit sphere
    W = np.random.randn(n, d)
    W_norm = np.linalg.norm(W, axis=1, keepdims=True)
    W_norm[W_norm == 0] = 1.0  # Prevent division by zero
    W = W / W_norm
    W_flat = W.flatten()

    # Precompute indices for the upper triangle of the Gram matrix
    i, j_idx = np.triu_indices(n, k=1)

    def objective(w_flat, alpha):
        W_curr = w_flat.reshape((n, d))

        # Safe norm calculation to avoid NaN gradients at the origin
        W_sq_norm = jnp.sum(W_curr**2, axis=1, keepdims=True)
        W_norm_curr = jnp.sqrt(jnp.maximum(W_sq_norm, 1e-12))

        # Project onto the sphere
        X = W_curr / W_norm_curr

        # Compute pairwise dot products (Gram Matrix)
        dot_products = jnp.dot(X, X.T)
        dp = dot_products[i, j_idx]

        # LogSumExp to smoothly approximate the maximum dot product
        lse = jax.scipy.special.logsumexp(alpha * dp) / alpha

        # Regularization to keep unnormalized vectors structurally close to the unit sphere
        reg = 1.0 * jnp.sum((W_norm_curr - 1.0) ** 2)

        return lse + reg

    # JIT compile the value and gradient computation to run at C/CUDA speeds
    val_grad = jit(value_and_grad(objective, argnums=0))

    # Continuation method schedule: (alpha, maxiter)
    stages = [
        (2.0, 300),
        (10.0, 300),
        (50.0, 300),
        (200.0, 300),
        (1000.0, 300),
        (4000.0, 500),
    ]

    for alpha, maxiter in stages:

        def fun(w):
            v, g = val_grad(w, alpha)
            return float(v), np.asarray(g, dtype=np.float64)

        # Optimize iteratively using L-BFGS-B
        res = minimize(
            fun,
            W_flat,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6},
        )
        W_flat = res.x

    # Final strict projection to the unit hypersphere
    W_final = W_flat.reshape((n, d))
    final_norms = np.linalg.norm(W_final, axis=1, keepdims=True)
    final_norms[final_norms == 0] = 1.0
    X_final = W_final / final_norms

    return X_final
