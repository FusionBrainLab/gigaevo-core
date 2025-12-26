import jax
import jax.numpy as jnp
import numpy as np


def _hermite_poly_recurrence(n: int, x: jnp.ndarray) -> jnp.ndarray:
    if n == 0:
        return jnp.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        h0 = jnp.ones_like(x)
        h1 = 2 * x
        for k in range(2, n + 1):
            h_new = 2 * x * h1 - 2 * (k - 1) * h0
            h0 = h1
            h1 = h_new
        return h1


def _hermite_4k_polys_values(m: int, x: jnp.ndarray) -> jnp.ndarray:
    degrees = [4 * k for k in range(m + 1)]
    H_vals = jnp.stack([_hermite_poly_recurrence(d, x) for d in degrees])
    return H_vals


def _construct_P_with_forced_zero_jax(coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    m = len(coeffs)
    H_vals = _hermite_4k_polys_values(m + 1, x)  
    
    partial = jnp.sum(coeffs[:, None] * H_vals[:m], axis=0)
    
    H_vals_at_zero = _hermite_4k_polys_values(m + 1, 0.0)
    partial_at_zero = jnp.sum(coeffs * H_vals_at_zero[:m])
    H_m_at_zero = H_vals_at_zero[m]
    c_last = -partial_at_zero / (H_m_at_zero + 1e-12)
    
    P = partial + c_last * H_vals[m]
    
    H_vals_large = _hermite_4k_polys_values(m + 1, 100.0)
    partial_large = jnp.sum(coeffs * H_vals_large[:m])
    P_large = partial_large + c_last * H_vals_large[m]
    sign_flip = jnp.where(P_large < 0, -1.0, 1.0)
    
    return P * sign_flip


def _largest_positive_root_of_P_over_x2_jax(coeffs: jnp.ndarray) -> jnp.ndarray:
    x_grid = jnp.linspace(0.01, 10.0, 2000)
    
    def P_div_x2(x):
        P_val = _construct_P_with_forced_zero_jax(coeffs, x)
        return P_val / (x**2 + 1e-12)
    
    P_div_x2_grid = jax.vmap(P_div_x2)(x_grid)
    
    signs = jnp.sign(P_div_x2_grid)
    diff_signs = jnp.diff(signs)
    sign_changes = jnp.where(diff_signs != 0)[0]

    def find_root_with_sign_changes():
        largest_idx = sign_changes[-1]
        x_guess = x_grid[largest_idx + 1]
        
        def f_scalar(x_val):
            P_val = _construct_P_with_forced_zero_jax(coeffs, jnp.array([x_val]))
            return P_val[0] / (x_val**2 + 1e-12)
        
        def newton_step(x):
            f_val = f_scalar(x)
            f_prime = jax.grad(f_scalar)(x)
            return x - f_val / (f_prime + 1e-12)
        
        x_current = x_guess
        def refine_iteration(i, x_curr):
            return newton_step(x_curr)
        
        x_refined = jax.lax.fori_loop(0, 50, refine_iteration, x_current)
        return x_refined
    
    def find_root_no_sign_changes():
        closest_idx = jnp.argmin(jnp.abs(P_div_x2_grid))
        return x_grid[closest_idx]
    

    has_sign_changes = len(sign_changes) > 0
    rmax = jax.lax.cond(
        has_sign_changes,
        find_root_with_sign_changes,
        find_root_no_sign_changes
    )
    
    return rmax


def compute_c(coefficients: jnp.ndarray) -> jnp.ndarray:
    coefficients = jnp.asarray(coefficients, dtype=jnp.float32)
    
    rmax = _largest_positive_root_of_P_over_x2_jax(coefficients)
    c = (rmax**2) / (2.0 * jnp.pi)
    
    return c
