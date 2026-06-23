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
            clamped_ips = jnp.clip(ips[i, j], -1.0 + 1e-12, 1.0 - 1e-12)
            geodesic_dists = jnp.arccos(clamped_ips)
            return jax.nn.logsumexp(-alpha * geodesic_dists) / alpha

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
        V /= np.linalg.norm(V, axis=1, keepdims=True)

        def compute_mu(X):
            ips = X @ X.T
            np.fill_diagonal(ips, -1.0)
            return float(np.max(ips))

        best_V = V.copy()
        best_mu = compute_mu(best_V)

        def make_callback():
            nonlocal best_V, best_mu

            def callback(xk):
                nonlocal best_V, best_mu
                v_reshaped = xk.reshape((self.n, self.d))
                norms = np.linalg.norm(v_reshaped, axis=1, keepdims=True)
                v_normed = v_reshaped / np.maximum(norms, 1e-8)
                current_mu = compute_mu(v_normed)
                if current_mu < best_mu:
                    best_mu = current_mu
                    best_V = v_normed.copy()

            return callback

        alphas = np.geomspace(20.0, 150000.0, num=8)
        maxiters = [200, 300, 400, 500, 800, 1000, 1200, 1500]
        for alpha, maxiter in zip(alphas, maxiters):

            def fun(v_flat):
                v_reshaped = v_flat.reshape((self.n, self.d))
                val, grad = self.val_grad_lse(v_reshaped, float(alpha))
                grad = np.array(grad, dtype=np.float64)
                norms = np.linalg.norm(v_reshaped, axis=1, keepdims=True)
                v_normed = v_reshaped / np.maximum(norms, 1e-8)
                grad_proj = (
                    grad - np.sum(grad * v_normed, axis=1, keepdims=True) * v_normed
                )
                return float(val), grad_proj.flatten()

            res = scipy.optimize.minimize(
                fun,
                V.flatten(),
                method="L-BFGS-B",
                jac=True,
                callback=make_callback(),
                options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-10},
            )
            V = res.x.reshape((self.n, self.d))
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            current_mu = compute_mu(V)
            if current_mu < best_mu:
                best_mu = current_mu
                best_V = V.copy()

        return best_V

    def perturb(self, points: np.ndarray, intensity: float, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        new_points = points.copy()
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)

        if intensity > 0.05:
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

        ips = np.dot(new_points, new_points.T)
        np.fill_diagonal(ips, -2.0)
        max_ip = np.max(ips)
        weights = np.exp((10.0 + 50.0 * intensity) * (ips - max_ip))
        np.fill_diagonal(weights, 0.0)

        repulsions = np.zeros_like(new_points)
        for i in range(self.n):
            diffs = (
                np.dot(new_points, new_points[i])[:, np.newaxis] * new_points[i]
                - new_points
            )
            repulsions[i] = np.sum(weights[i][:, np.newaxis] * diffs, axis=0)

        for i in range(self.n):
            if np.any(repulsions[i]):
                direction = repulsions[i]
                direction -= np.dot(direction, new_points[i]) * new_points[i]
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-8:
                    direction /= norm_dir
                    step_size = rng.uniform(0.01, 0.1) * intensity
                    new_points[i] = new_points[i] * np.cos(
                        step_size
                    ) + direction * np.sin(step_size)

        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)

        num_destroy = max(1, int(self.n * 0.15 * intensity))
        ips_temp = np.dot(new_points, new_points.T)
        np.fill_diagonal(ips_temp, -2.0)
        max_ips_per_node = np.max(ips_temp, axis=1)
        worst_nodes = np.argsort(max_ips_per_node)[::-1]
        destroy_idx = worst_nodes[:num_destroy]
        keep_idx = worst_nodes[num_destroy:]

        num_candidates = max(200, self.n * 5)
        candidates = rng.standard_normal((num_candidates, self.d))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)

        selected_points = list(new_points[keep_idx])
        for _ in range(num_destroy):
            current_points = np.array(selected_points)
            ips_candidates = np.dot(candidates, current_points.T)
            max_ips = np.max(ips_candidates, axis=1)
            best_idx = np.argmin(max_ips)
            selected_points.append(candidates[best_idx])
            candidates = np.delete(candidates, best_idx, axis=0)

        new_points[destroy_idx] = np.array(selected_points[-num_destroy:])
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)

        s_val = float(self.d) + 5.0 * (1.0 - intensity)

        def fun_s(v_flat):
            v_reshaped = v_flat.reshape((self.n, self.d))
            val, grad = self.val_grad_s_energy(v_reshaped, float(s_val))
            grad = np.array(grad, dtype=np.float64)
            norms = np.linalg.norm(v_reshaped, axis=1, keepdims=True)
            v_normed = v_reshaped / np.maximum(norms, 1e-8)
            grad_proj = grad - np.sum(grad * v_normed, axis=1, keepdims=True) * v_normed
            return float(val), grad_proj.flatten()

        res = scipy.optimize.minimize(
            fun_s,
            new_points.flatten(),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(100 + 150 * intensity)},
        )
        new_points = res.x.reshape((self.n, self.d))
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        return new_points


def entrypoint():
    return Improver
