import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

@jax.jit
def get_unit_hex_verts(center, theta):
    # Regular hexagon vertices, circumradius 1.0
    base_angles = jnp.array([0.0, jnp.pi/3, 2*jnp.pi/3, jnp.pi, 4*jnp.pi/3, 5*jnp.pi/3], dtype=jnp.float64)
    angles = base_angles + theta
    return center + jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

@jax.jit
def get_container_normals():
    # Normals for flat-topped enclosing hexagon (face-centered directions)
    angles = jnp.array([jnp.pi/6, jnp.pi/2, 5*jnp.pi/6, 7*jnp.pi/6, 3*jnp.pi/2, 11*jnp.pi/6], dtype=jnp.float64)
    return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

@jax.jit
def compute_separation(c1, theta1, c2, theta2):
    v1 = get_unit_hex_verts(c1, theta1)
    v2 = get_unit_hex_verts(c2, theta2)
    axes_angles = jnp.array([jnp.pi/6, jnp.pi/2, 5*jnp.pi/6], dtype=jnp.float64)
    n1 = jnp.stack([jnp.cos(axes_angles + theta1), jnp.sin(axes_angles + theta1)], axis=1)
    n2 = jnp.stack([jnp.cos(axes_angles + theta2), jnp.sin(axes_angles + theta2)], axis=1)
    axes = jnp.concatenate([n1, n2], axis=0)
    p1 = jnp.dot(v1, axes.T)
    p2 = jnp.dot(v2, axes.T)
    gaps = jnp.maximum(jnp.min(p1, axis=0) - jnp.max(p2, axis=0), jnp.min(p2, axis=0) - jnp.max(p1, axis=0))
    return jnp.max(gaps)

@jax.jit
def objective_fn(params, idx_i, idx_j, weight_cont, weight_ov, margin):
    N = (len(params) - 1) // 3
    centers = params[:2*N].reshape((N, 2))
    thetas = params[2*N:3*N]
    L = params[-1]
    
    # Flat-topped container containment
    H = L * (jnp.sqrt(3.0) / 2.0)
    normals = get_container_normals()
    
    def hex_cont(c, t):
        v = get_unit_hex_verts(c, t)
        return jnp.sum(jax.nn.relu(jnp.dot(v, normals.T) - H)**2)
    
    total_cont = jnp.sum(jax.vmap(hex_cont)(centers, thetas))

    def hex_ov(i, j):
        sep = compute_separation(centers[i], thetas[i], centers[j], thetas[j])
        return jax.nn.relu(margin - sep)**2

    total_ov = jnp.sum(jax.vmap(hex_ov)(idx_i, idx_j))
    
    return L + weight_cont * total_cont + weight_ov * total_ov

_grad_jit = jax.jit(jax.value_and_grad(objective_fn))

class Improver:
    def __init__(self, hex_num=11, seed: int = 0):
        self.hex_num = hex_num
        self.seed = seed
        self.idx_i, self.idx_j = map(jnp.array, np.triu_indices(hex_num, k=1))

    def generate_config(self, seed=None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        centers = []
        for q in range(-5, 6):
            for r in range(-5, 6):
                if abs(q+r) <= 5:
                    # Hexagonal lattice base
                    x = (q + r/2.0) * np.sqrt(3)
                    y = r * 1.5
                    centers.append([x, y])
        centers = np.array(centers)
        dists = np.linalg.norm(centers, axis=1)
        # Start with slightly jittered central points
        idx = np.argsort(dists + rng.uniform(0, 0.5, size=len(dists)))[:self.hex_num]
        return centers[idx], rng.uniform(0, np.pi/3, self.hex_num)

    def perturb(self, input_config, intensity, seed=None):
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        c, a = map(np.copy, input_config)
        if intensity > 2.0:
            # Structural swap to escape local optima
            i, j = rng.choice(self.hex_num, 2, replace=False)
            c[i], c[j] = c[j].copy(), c[i].copy()
            a[i], a[j] = a[j].copy(), a[i].copy()
        
        c += rng.normal(0, 0.05 * intensity, c.shape)
        a += rng.normal(0, 0.02 * intensity, a.shape)
        return c, np.mod(a, np.pi/3)

    def improve(self, input_config, seed=None):
        centers, angles = input_config
        L_est = np.max(np.linalg.norm(centers, axis=1)) + 1.0
        x = np.concatenate([centers.flatten(), angles, [L_est]]).astype(np.float64)
        
        # Graduated hardening schedule for N=11 precision
        stages = [
            {'w_cont': 1e3, 'w_ov': 1e2, 'margin': 1e-3},
            {'w_cont': 1e4, 'w_ov': 1e4, 'margin': 1e-4},
            {'w_cont': 1e6, 'w_ov': 1e6, 'margin': 1e-5},
            {'w_cont': 1e9, 'w_ov': 1e8, 'margin': 1e-7}
        ]

        for st in stages:
            res = minimize(
                lambda p: tuple(map(np.array, _grad_jit(p, self.idx_i, self.idx_j, st['w_cont'], st['w_ov'], st['margin']))),
                x, method='L-BFGS-B', jac=True,
                bounds=[(-15, 15)]*(2*self.hex_num) + [(0.0, np.pi/3)]*self.hex_num + [(1.0, 15.0)],
                options={'maxiter': 800, 'ftol': 1e-10}

            )
            x = res.x

        final_centers = x[:2*self.hex_num].reshape((self.hex_num, 2))
        final_angles = np.mod(x[2*self.hex_num:3*self.hex_num], 2*np.pi)
        return final_centers, final_angles

def entrypoint():
    return Improver