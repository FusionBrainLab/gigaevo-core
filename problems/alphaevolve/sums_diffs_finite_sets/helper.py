import jax
import jax.numpy as jnp


def compute_c6(u_set: jnp.ndarray) -> float:
    U = jnp.sort(u_set)

    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]

    size_U_plus_U = jnp.unique(sums).shape[0]
    size_U_minus_U = jnp.unique(diffs).shape[0]
    max_U = jnp.max(U)

    eps = 1e-9
    max_U_safe = jnp.maximum(max_U, eps)

    ratio = size_U_minus_U / (size_U_plus_U + eps)
    log_ratio = jnp.log(ratio)
    log_denom = jnp.log(2 * max_U_safe + 1)

    c6_bound = 1 + log_ratio / log_denom

    return c6_bound
