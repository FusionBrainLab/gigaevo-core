import numpy as np
import jax
import jax.numpy as jnp
import optax

from helper import get_matrix_multiplication_tensor

n, m, p = 2, 4, 5


@jax.custom_vjp
def round_to_half_ste(x):
    return jnp.round(x * 2) / 2


def round_ste_fwd(x):
    return round_to_half_ste(x), None


def round_ste_bwd(res, g):
    return (g,)


round_to_half_ste.defvjp(round_ste_fwd, round_ste_bwd)


def weighted_l2_loss(reconstructed: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    error = reconstructed - target
    weights = jnp.where(target != 0, 100.0, 1.0)
    return jnp.mean(weights * (error**2))


def l2_loss_real(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((x - y) ** 2)


def get_constrained_decomposition(latent_decomposition: tuple, clamp_range: float) -> tuple:
    return jax.tree_util.tree_map(
        lambda x: clamp_range * jnp.tanh(x), latent_decomposition
    )


@jax.jit
def train_step(params, opt_state, optimizer, loss_fn):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def entrypoint() -> dict:
    rank = 55
    num_restarts = 10
    phase1_steps = 80000
    phase1_lr = 0.01
    init_scale = 0.1
    l1_strength = 1e-6
    clamp_range = 4.0
    phase2_steps = 20000
    phase2_lr = 1e-4
    
    target_tensor = get_matrix_multiplication_tensor(n, m, p)
    main_key = jax.random.PRNGKey(42)
    
    def phase1_loss_fn(latent_decomposition: tuple) -> jnp.ndarray:
        constrained = get_constrained_decomposition(latent_decomposition, clamp_range)
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *constrained)
        recon_loss = weighted_l2_loss(reconstructed, target_tensor)
        l1_penalty = sum(jnp.mean(jnp.abs(arr)) for arr in constrained)
        return recon_loss + l1_strength * l1_penalty
    
    def phase2_loss_fn(continuous_decomposition: tuple) -> jnp.ndarray:
        discrete_decomposition = jax.tree_util.tree_map(round_to_half_ste, continuous_decomposition)
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *discrete_decomposition)
        return l2_loss_real(reconstructed, target_tensor)
    
    best_loss_phase1 = float("inf")
    best_latent_decomp = None
    phase1_optimizer = optax.adam(phase1_lr)
    
    for i in range(num_restarts):
        main_key, restart_key = jax.random.split(main_key)
        init_fn = jax.nn.initializers.normal(stddev=init_scale)
        latent_decomp = (
            init_fn(restart_key, (n * m, rank)),
            init_fn(restart_key, (m * p, rank)),
            init_fn(restart_key, (n * p, rank)),
        )
        opt_state = phase1_optimizer.init(latent_decomp)
        
        for _ in range(phase1_steps):
            latent_decomp, opt_state, loss = train_step(
                latent_decomp,
                opt_state,
                phase1_optimizer,
                phase1_loss_fn,
            )
        
        final_loss = l2_loss_real(
            target_tensor,
            jnp.einsum(
                "ir,jr,kr->ijk",
                *get_constrained_decomposition(latent_decomp, clamp_range),
            ),
        )
        
        if final_loss < best_loss_phase1:
            best_loss_phase1 = final_loss
            best_latent_decomp = latent_decomp
    
    continuous_params = get_constrained_decomposition(best_latent_decomp, clamp_range)
    phase2_optimizer = optax.adam(phase2_lr)
    opt_state = phase2_optimizer.init(continuous_params)
    
    for step in range(phase2_steps):
        continuous_params, opt_state, loss = train_step(
            continuous_params, opt_state, phase2_optimizer, phase2_loss_fn
        )
        if loss < 1e-7:
            break
    
    final_discrete_decomposition = jax.tree_util.tree_map(round_to_half_ste, continuous_params)
    final_decomposition_np = jax.tree_util.tree_map(np.array, final_discrete_decomposition)
    u_reshaped, v_reshaped, w_reshaped = final_decomposition_np
    
    u_vectors = u_reshaped.T
    v_vectors = v_reshaped.T
    w_vectors = w_reshaped.T
    
    return {
        "rank": rank,
        "u_vectors": u_vectors,
        "v_vectors": v_vectors,
        "w_vectors": w_vectors,
    }

