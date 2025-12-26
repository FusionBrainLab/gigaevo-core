import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy.special import hermite

from helper import compute_c


def entrypoint() -> np.ndarray:
    learning_rate = 0.001
    num_steps = 100000
    num_restarts = 20
    num_hermite_coeffs = 4
    
    degrees = [4 * k for k in range(num_hermite_coeffs)]
    max_degree = degrees[-1]
    hermite_polys = [hermite(d) for d in degrees]
    
    hermite_basis = []
    for poly in hermite_polys:
        pad_amount = max_degree - poly.order
        hermite_basis.append(jnp.array(np.pad(poly.coef, (pad_amount, 0))))
    hermite_basis = jnp.stack(hermite_basis)
    
    H_vals_at_zero = jnp.array([p(0) for p in hermite_polys])
    x_grid = jnp.linspace(0.0, 10.0, 3000)
    optimizer = optax.adam(learning_rate)
    
    def objective_fn(params: jnp.ndarray):
        c_others, log_c_last = params[:-1], params[-1]
        c_last = jnp.exp(log_c_last)
        
        c0 = (
            -(jnp.sum(c_others * H_vals_at_zero[1:-1]) + c_last * H_vals_at_zero[-1])
            / H_vals_at_zero[0]
        )
        hermite_coeffs = jnp.concatenate([jnp.array([c0]), c_others, jnp.array([c_last])])
        
        poly_coeffs_std = jnp.sum(hermite_coeffs[:, None] * hermite_basis, axis=0)
        p_values = jnp.polyval(poly_coeffs_std, x_grid)
        
        weights = 1.0 + (x_grid / (x_grid[-1] + 1e-12))
        loss = jnp.sum(weights * jax.nn.relu(-p_values))
        return loss
    
    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(objective_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    def run_single_trial(key: jax.random.PRNGKey):
        num_params_to_opt = num_hermite_coeffs - 1
        assert num_params_to_opt == 3, "This initialization assumes num_hermite_coeffs == 4."
        
        base_c1 = -0.01158510802599293
        base_c2 = -8.921606035407065e-05
        base_log_c_last = np.log(1e-6)
        
        base = jnp.array([base_c1, base_c2, base_log_c_last], dtype=jnp.float32)
        noise = jax.random.normal(key, (num_params_to_opt,)) * 1e-3
        params = base + noise
        
        opt_state = optimizer.init(params)
        
        for _ in range(num_steps):
            params, opt_state, _ = train_step(params, opt_state)
        return params
    
    def get_coeffs_from_params(params: np.ndarray):
        c_others, log_c_last = params[:-1], params[-1]
        c_last = np.exp(log_c_last)
        
        c0 = (
            -(np.sum(c_others * H_vals_at_zero[1:-1]) + c_last * H_vals_at_zero[-1]) / H_vals_at_zero[0]
        )
        hermite_coeffs = np.concatenate([[c0], np.array(c_others), [c_last]])
        return jnp.array(hermite_coeffs, dtype=jnp.float32)
    
    main_key = jax.random.PRNGKey(42)
    best_c = float("inf")
    best_coeffs = None
    
    for _ in range(num_restarts):
        main_key, restart_key = jax.random.split(main_key)
        final_params = run_single_trial(restart_key)
        
        coeffs = get_coeffs_from_params(np.array(final_params))
        c = compute_c(coeffs)
        
        if c < best_c:
            best_c = float(c)
            best_coeffs = coeffs
    
    if best_coeffs is None:
        raise RuntimeError("Failed to find a valid solution in any restart.")
    
    return np.array(best_coeffs[:num_hermite_coeffs])

