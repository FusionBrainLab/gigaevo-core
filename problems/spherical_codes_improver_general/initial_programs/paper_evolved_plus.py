"""D-dimensional spherical-code improver — ImprovEvolve+E LLM-evolved program (E.8).

The stronger variant from the paper (Appendix E.8): a softer/longer LogSumExp
schedule (7 alphas, more iterations), upper-triangular surrogates, and a perturb
that accumulates concurrent active-set repulsions into tangent-projected geodesic
steps, adds structured tangent noise, relaxes an s-energy, then applies a
hemispheric "tectonic twist" rotation at higher intensity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.optimize

jax.config.update("jax_enable_x64", True)


class Improver:
    def __init__(self, n: int, d: int, seed: int = 0):
        self.n = n
        self.d = d
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        n_static = self.n

        @jax.jit
        def objective_lse(V, alpha):
            norms = jnp.linalg.norm(V, axis=1, keepdims=True)
            X = V / jnp.maximum(norms, 1e-8)
            ips = jnp.dot(X, X.T)
            i, j = jnp.triu_indices(n_static, k=1)
            return jax.nn.logsumexp(alpha * ips[i, j]) / alpha

        self.val_grad_lse = jax.jit(jax.value_and_grad(objective_lse, argnums=0))

        @jax.jit
        def s_energy(V, s):
            norms = jnp.linalg.norm(V, axis=1, keepdims=True)
            X = V / jnp.maximum(norms, 1e-8)
            ips = jnp.dot(X, X.T)
            i, j = jnp.triu_indices(n_static, k=1)
            dists_sq = jnp.maximum(2.0 - 2.0 * ips[i, j], 1e-10)
            return jnp.sum(1.0 / jnp.power(dists_sq, s / 2.0))

        self.val_grad_s_energy = jax.jit(jax.value_and_grad(s_energy, argnums=0))

    def generate_config(self, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        points = rng.standard_normal((self.n, self.d))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return points

    def improve(self, points: np.ndarray, seed=None) -> np.ndarray:
        V = points.copy()
        alphas = [10.0, 40.0, 160.0, 640.0, 2560.0, 10240.0, 40960.0]
        for alpha in alphas:
            def fun(v_flat):
                v_reshaped = v_flat.reshape((self.n, self.d))
                val, grad = self.val_grad_lse(v_reshaped, float(alpha))
                return float(val), np.array(grad, dtype=np.float64).flatten()

            res = scipy.optimize.minimize(
                fun,
                V.flatten(),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-10},
            )
            V = res.x.reshape((self.n, self.d))
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        return V / norms

    def perturb(self, points: np.ndarray, intensity: float, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        new_points = points.copy()
        ips = np.dot(new_points, new_points.T)
        np.fill_diagonal(ips, -2.0)
        max_ip = np.max(ips)
        threshold = max_ip - 0.02 * (1.0 + intensity)
        active_mask = ips > threshold

        repulsions = np.zeros_like(new_points)
        for i in range(self.n):
            neighbors = new_points[active_mask[i]]
            if len(neighbors) > 0:
                repulsions[i] = np.sum(new_points[i] - neighbors, axis=0)

        for i in range(self.n):
            if np.any(repulsions[i]):
                direction = repulsions[i]
                direction -= np.dot(direction, new_points[i]) * new_points[i]
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-8:
                    direction /= norm_dir
                    step_size = rng.uniform(0.01, 0.1) * intensity
                    new_points[i] = new_points[i] * np.cos(step_size) + direction * np.sin(step_size)

        noise = rng.standard_normal((self.n, self.d))
        for i in range(self.n):
            n_vec = noise[i] - np.dot(noise[i], new_points[i]) * new_points[i]
            n_norm = np.linalg.norm(n_vec)
            if n_norm > 1e-8:
                new_points[i] += rng.uniform(0.005, 0.02) * (intensity + 0.1) * (n_vec / n_norm)

        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        s_val = float(self.d) + 20.0 * (1.0 - intensity)

        def fun_s(v_flat):
            v_reshaped = v_flat.reshape((self.n, self.d))
            val, grad = self.val_grad_s_energy(v_reshaped, float(s_val))
            return float(val), np.array(grad, dtype=np.float64).flatten()

        res = scipy.optimize.minimize(
            fun_s,
            new_points.flatten(),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(5 + 15 * intensity)},
        )
        new_points = res.x.reshape((self.n, self.d))
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)

        if intensity > 0.3:
            split_v = rng.standard_normal(self.d)
            split_v /= np.linalg.norm(split_v)
            subset_indices = np.where(np.dot(new_points, split_v) > 0)[0]
            if len(subset_indices) > 0:
                U_random = rng.standard_normal((self.d, self.d))
                Q, _ = np.linalg.qr(U_random)
                A = Q - Q.T
                A /= np.linalg.norm(A) + 1e-8
                theta = rng.uniform(0.1, 0.5) * intensity
                R = scipy.linalg.expm(theta * A)
                new_points[subset_indices] = np.dot(new_points[subset_indices], R.T)

        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        return new_points


def entrypoint():
    return Improver
