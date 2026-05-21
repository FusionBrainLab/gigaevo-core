import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


def entrypoint(n=600, d=11, seed=0) -> np.ndarray:
    """
    Calculates an optimized spherical code configuration by minimizing the
    maximum pairwise cosine similarity (inner product) among points on the
    unit sphere.

    Algorithm:
    We frame the problem as an unconstrained optimization over \(\mathbb{R}^{n \times d}\).
    To approximate the minimax objective \(\min_X \max_{i < j} \langle x_i, x_j \rangle\),
    we minimize a sequence of differentiable LogSumExp functions:

    \[ L_\tau(X) = \frac{1}{\tau} \log \sum_{i < j} \exp\left(\tau \langle \tilde{x}_i, \tilde{x}_j \rangle \right) + \frac{1}{2} \sum_{i=1}^n (\|\tilde{x}_i\| - 1)^2 \]

    where \(\tilde{x}_i = x_i / \max(\|x_i\|_2, \epsilon)\) represents the row-normalized points.
    We progressively increase the sharpness parameter \(\tau\) over several
    optimization phases using L-BFGS-B to accurately refine the configuration.
    The regularization ensures that coordinates do not numerically drift away
    from the unit sphere scale.

    Args:
        n: Number of points.
        d: Dimension of the space (points lie on S^{d-1} in R^d).
        seed: Random seed for reproducibility.

    Returns:
        points: np.ndarray of shape (n, d) representing the optimized configuration.
    """
    # Enable float64 for better precision during numerical optimization
    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(seed)

    # Initialize points uniformly on the sphere using Gaussian generation
    X_init = rng.normal(size=(n, d))
    X_init = X_init / np.linalg.norm(X_init, axis=1, keepdims=True)

    # Precompute indices for the upper triangle (distinct pairs)
    # to avoid the diagonal and redundant pairs during objective calculation
    row_idx, col_idx = np.triu_indices(n, k=1)
    row_idx = jnp.array(row_idx)
    col_idx = jnp.array(col_idx)

    def loss_fn(x_flat, tau):
        X = x_flat.reshape((n, d))

        # Smoothly compute norms to avoid NaN gradients at exactly zero
        sq_norms = jnp.sum(X**2, axis=1, keepdims=True)
        norms = jnp.sqrt(sq_norms + 1e-16)

        # Implicitly project points onto the unit sphere
        X_norm = X / norms

        # Compute all pairwise cosine similarities
        S = X_norm @ X_norm.T

        # Extract distinct pairs
        S_upper = S[row_idx, col_idx]

        # LogSumExp smooth maximum approximation
        logsumexp_loss = jax.nn.logsumexp(tau * S_upper) / tau

        # Regularization keeping the latent coordinates near norm 1
        # to prevent optimization drift across scale-invariant flat spaces
        reg_loss = 0.5 * jnp.mean((norms.squeeze() - 1.0) ** 2)

        return logsumexp_loss + reg_loss

    # JIT compile the value and gradient computations for efficiency
    val_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    def scipy_obj(x_flat, tau):
        # SciPy expects strictly scalar float and float64 arrays
        loss, grad = val_and_grad_fn(x_flat, tau)
        return float(loss), np.array(grad, dtype=np.float64)

    x_opt = X_init.flatten()

    # Schedule of tau values to gradually transform the objective into
    # a strict maximum function. Increasing tau slowly prevents getting
    # overly trapped in poor local minima early on.
    taus = [10.0, 50.0, 200.0, 1000.0]
    max_iters = [500, 500, 1000, 1000]

    for tau, max_iter in zip(taus, max_iters):
        res = minimize(
            scipy_obj,
            x_opt,
            args=(tau,),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": 1e-7, "gtol": 1e-5},
        )
        x_opt = res.x

    # Final strict normalization to flawlessly satisfy geometric constraints
    X_final = x_opt.reshape((n, d))
    X_final = X_final / np.linalg.norm(X_final, axis=1, keepdims=True)

    return X_final
