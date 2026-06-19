"""D-dimensional spherical-code improver — ImprovEvolve LLM-evolved program (E.7).

Verbatim reconstruction of the Gemini-3.5-Flash-evolved Improver from the paper
(Appendix E.7), the published baseline for the dimensionality scan. improve runs
a LogSumExp continuation on the max-cosine objective with L-BFGS-B; perturb does
active-set repulsion at the min-distance pairs, an s-energy relaxation, and a
global subset rotation, keyed off intensity.
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
            idx = jnp.arange(n_static)
            ips_masked = ips.at[idx, idx].set(-jnp.inf)
            max_ip = jnp.max(ips_masked)
            sum_exp = jnp.sum(jnp.exp(alpha * (ips_masked - max_ip)))
            return max_ip + jnp.log(sum_exp) / alpha

        self.val_grad_lse = jax.jit(jax.value_and_grad(objective_lse, argnums=0))

        @jax.jit
        def s_energy(V, s):
            norms = jnp.linalg.norm(V, axis=1, keepdims=True)
            X = V / jnp.maximum(norms, 1e-8)
            ips = jnp.dot(X, X.T)
            dists_sq = jnp.maximum(2.0 - 2.0 * ips, 1e-10)
            idx = jnp.arange(n_static)
            dists_sq = dists_sq.at[idx, idx].set(jnp.inf)
            return jnp.sum(1.0 / jnp.power(dists_sq, s / 2.0))

        self.val_grad_s_energy = jax.jit(jax.value_and_grad(s_energy, argnums=0))

    def generate_config(self, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        points = rng.standard_normal((self.n, self.d))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return points

    def improve(self, points: np.ndarray, seed=None) -> np.ndarray:
        V = points.copy()
        alphas = [100.0, 400.0, 1600.0, 6400.0, 25600.0]
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
                options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9},
            )
            V = res.x.reshape((self.n, self.d))
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        return V / norms

    def perturb(self, points: np.ndarray, intensity: float, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        new_points = points.copy()
        ips = np.dot(new_points, new_points.T)
        np.fill_diagonal(ips, -np.inf)
        max_ip = np.max(ips)
        threshold = max_ip - 0.005 * (1.0 + intensity)
        active_pairs = np.argwhere(ips > threshold)

        for i, j in active_pairs:
            if i > j:
                continue
            direction = new_points[i] - new_points[j]
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-8:
                direction /= norm_dir
                step_size = rng.uniform(0.01, 0.05) * (intensity + 0.1)
                new_points[i] += step_size * direction
                new_points[j] -= step_size * direction
            for idx in (i, j):
                noise = rng.standard_normal(self.d)
                noise -= np.dot(noise, new_points[idx]) * new_points[idx]
                new_points[idx] += (
                    rng.uniform(0.005, 0.02)
                    * (intensity + 0.1)
                    * (noise / np.linalg.norm(noise))
                )

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
            num_to_rotate = int(self.n * rng.uniform(0.1, 0.5 * intensity))
            if num_to_rotate > 0:
                subset_indices = rng.choice(self.n, num_to_rotate, replace=False)
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
