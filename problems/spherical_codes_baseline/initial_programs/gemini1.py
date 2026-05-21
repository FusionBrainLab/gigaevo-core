import numpy as np


def entrypoint(n=600, d=11, seed=0) -> np.ndarray:
    """
    Calculates the optimal spherical code configuration.

    Algorithm Explanation:
    To find a set of points $X = \{x_1, \dots, x_n\}$ on the unit hypersphere $S^{d-1}$
    that minimizes the maximum pairwise cosine similarity $\mu(X) = \max_{i < j} x_i \cdot x_j$,
    we solve a continuous optimization problem. The non-smooth minimax objective is relaxed
    using the smooth Log-Sum-Exp (LSE) approximation:
    $$ \text{LSE}_\tau(S) = \frac{1}{\tau} \log \sum_{i < j} \exp(\tau \cdot (x_i \cdot x_j)) $$
    where $\tau > 0$ is a temperature parameter. As $\tau \to \infty$, the LSE converges
    to the exact maximum inner product.

    We parameterize the configuration with unconstrained vectors $V \in \mathbb{R}^{n \times d}$,
    obtaining points on the hypersphere via projection: $x_i = v_i / \|v_i\|_2$.
    We optimize the LSE objective using the Adam optimizer with a cosine decay learning
    rate schedule. The temperature $\tau$ is linearly increased during the optimization
    (a cooling schedule) to accurately target the true maximum pairwise inner product
    as the points separate. A soft regularizer ensures $V$ remains near the unit sphere,
    preventing vanishing gradients or numerical drift.

    To ensure a highly optimal configuration and avoid poor local minima, we employ
    JAX's `vmap` to run multiple parallel optimization restarts simultaneously, finally
    returning the configuration that yields the absolute lowest maximum inner product.
    """
    import jax

    # Enable 64-bit precision to strictly guarantee unit vector requirements
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax

    rng = jax.random.PRNGKey(seed)

    # Precompute mask for strictly upper triangular part to ignore self-similarity
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)

    def loss(V, tau):
        # Add a tiny epsilon to prevent division by zero or NaN gradients
        norms = jnp.sqrt(jnp.sum(V**2, axis=-1, keepdims=True) + 1e-12)
        X = V / norms

        # Complete pairwise inner products
        S = jnp.dot(X, X.T)

        # Mask out diagonal and lower triangle elements by setting them to -infinity
        S_masked = jnp.where(mask, S, -jnp.inf)

        # Smooth maximum over the valid upper-triangle elements
        lse = jax.scipy.special.logsumexp(tau * S_masked) / tau

        # Soft regularization to anchor unconstrained vectors V near unit length
        reg = 1e-4 * jnp.mean((norms - 1.0) ** 2)
        return lse + reg

    steps = 10000
    # Decaying learning rate to settle into an optimal exact configuration
    scheduler = optax.cosine_decay_schedule(
        init_value=0.05, decay_steps=steps, alpha=0.001
    )
    optimizer = optax.adam(learning_rate=scheduler)

    # Temperature schedule: smooth transition from repulsive force to strict max-margin
    taus = jnp.linspace(10.0, 1000.0, steps)

    @jax.jit
    def optimize_multi(V_init_batch, taus):
        def single_optimize(V_init):
            opt_state = optimizer.init(V_init)

            # Utilizing scan for extremely fast bounded loops directly in XLA
            def scan_fn(state, tau):
                V, opt_state = state
                loss_val, grads = jax.value_and_grad(loss)(V, tau)
                updates, opt_state = optimizer.update(grads, opt_state, V)
                V = optax.apply_updates(V, updates)
                return (V, opt_state), None

            (V_final, _), _ = jax.lax.scan(scan_fn, (V_init, opt_state), taus)
            return V_final

        # Map the single optimization loop over our parallel batch of initializations
        return jax.vmap(single_optimize)(V_init_batch)

    # Generate 5 completely distinct random restarts
    num_restarts = 5
    rngs = jax.random.split(rng, num_restarts)
    V_inits = jax.vmap(lambda key: jax.random.normal(key, (n, d)))(rngs)

    # Execute the batched optimizations
    V_finals = optimize_multi(V_inits, taus)

    @jax.jit
    def max_dot_fn(V):
        X = V / jnp.linalg.norm(V, axis=-1, keepdims=True)
        S = jnp.dot(X, X.T)
        S_masked = jnp.where(mask, S, -jnp.inf)
        return jnp.max(S_masked)

    # Discern which initialization culminated in the best true spherical code
    max_dots = jax.vmap(max_dot_fn)(V_finals)
    best_idx = jnp.argmin(max_dots)
    best_V = V_finals[best_idx]

    # Ensure final absolute projection tightly on the hypersphere
    X_final = best_V / jnp.linalg.norm(best_V, axis=-1, keepdims=True)
    return np.array(X_final)
