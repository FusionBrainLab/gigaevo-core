r"""
=============================================================================
D-DIMENSIONAL SPHERICAL CODE OPTIMIZATION
=============================================================================

Methodology:
We frame the spherical code problem as a continuous min-max optimization
problem over the unit sphere $S^{d-1}$:
    \min_{X \in \mathbb{R}^{n \times d}} \max_{i \neq j} \frac{x_i \cdot x_j}{\|x_i\| \|x_j\|}

Since the true max function is non-smooth and prone to dead gradients, we
employ a smooth surrogate: the LogSumExp (LSE) function parameterized by
a temperature parameter \alpha.
    f_\alpha(X) = \mu + \frac{1}{\alpha} \log \sum_{i \neq j} \exp(\alpha (x_i \cdot x_j - \mu))
where \mu = \max_{i \neq j} x_i \cdot x_j.

Optimization ('improve' method):
We calculate the exact gradients of f_\alpha using JAX's reverse-mode
autodifferentiation. We perform unconstrained optimization using SciPy's
L-BFGS-B on the ambient coordinates V, where the objective function dynamically
projects X = V / ||V||. Because the objective only depends on directions,
the gradient is strictly orthogonal to V, naturally preventing norm collapse.
We apply a continuation method over \alpha (e.g., \alpha \in \{50, 100, 200, 400, 800\})
to systematically morph the smooth approximation toward the true max function.

Perturbation ('perturb' method):
The energy landscape of spherical codes is fraught with interlocking local minima.
Our `perturb` method orchestrates three heuristics, scaled by 'intensity':
1. Active-Set Targeting (Micro): We identify specific pairs (x_i, x_j) near
   the maximum inner product bottleneck. We inject targeted repulsion and
   orthogonal tangent-space noise exclusively to these bottlenecks.
2. Surrogate Relaxations (Meso/Macro): We temporarily optimize a Riesz
   s-energy surrogate using L-BFGS-B to smoothly deform rigid configurations.
3. Subspace Rotations (Macro): Employs Lie group exponentials (via skew-symmetric
   matrices) to generate small random SO(d) rotations applied solely to localized
   spatial subsets, preserving inner sub-structures while breaking global symmetries.
=============================================================================
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.optimize

# Ensure float64 precision for geometric stability and prevent numerical explosion
jax.config.update("jax_enable_x64", True)


class Improver:
    def __init__(self, n: int, d: int, seed: int = 0):
        """
        Args:
            n: Number of points.
            d: Dimension of the space.
            seed: Random seed.
        """
        self.n = n
        self.d = d
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        n_static = self.n

        # -----------------------------------------------------------
        # JAX JIT-Compiled Objectives
        # -----------------------------------------------------------
        @jax.jit
        def objective_lse(V, alpha):
            """Smooth LogSumExp surrogate for maximum cosine similarity."""
            norms = jnp.linalg.norm(V, axis=1, keepdims=True)
            # Safe projection
            X = V / jnp.maximum(norms, 1e-12)
            ips = jnp.dot(X, X.T)

            # Mask out the diagonal
            idx = jnp.arange(n_static)
            ips_masked = ips.at[idx, idx].set(-jnp.inf)

            max_ip = jnp.max(ips_masked)

            # LSE over all non-diagonal elements
            sum_exp = jnp.sum(jnp.exp(alpha * (ips_masked - max_ip)))
            return max_ip + jnp.log(sum_exp) / alpha

        # Function returning Value and Gradient for the LSE objective
        self.val_grad_lse = jax.jit(jax.value_and_grad(objective_lse, argnums=0))

        @jax.jit
        def s_energy(V, s):
            """Riesz s-energy surrogate used for landscape perturbation."""
            norms = jnp.linalg.norm(V, axis=1, keepdims=True)
            X = V / jnp.maximum(norms, 1e-12)
            ips = jnp.dot(X, X.T)

            dists_sq = 2.0 - 2.0 * ips

            # Mask out the diagonal
            idx = jnp.arange(n_static)
            dists_sq = dists_sq.at[idx, idx].set(jnp.inf)
            dists_sq = jnp.maximum(dists_sq, 1e-12)

            return jnp.sum(1.0 / jnp.power(dists_sq, s / 2.0))

        # Function returning Value and Gradient for the s-energy surrogate
        self.val_grad_s_energy = jax.jit(jax.value_and_grad(s_energy, argnums=0))

    def generate_config(self, seed=None) -> np.ndarray:
        """
        Generates a valid initial configuration of n points on the unit sphere.
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        points = rng.standard_normal((self.n, self.d))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return points

    def improve(self, points: np.ndarray, seed=None) -> np.ndarray:
        """
        Refines an existing configuration to minimize the maximum inner product
        via a rigorous continuation method using L-BFGS-B and JAX autodiff.
        """
        V = points.copy()

        # Continuation trajectory: progressively tighten the smooth max approximation
        alphas = [50.0, 100.0, 200.0, 400.0, 800.0]

        for alpha in alphas:

            def fun(v_flat):
                v_reshaped = v_flat.reshape((self.n, self.d))
                val, grad = self.val_grad_lse(v_reshaped, float(alpha))
                # Cast the returned gradient strictly to Python float64 array for SciPy L-BFGS-B
                return float(val), np.array(grad, dtype=np.float64).flatten()

            res = scipy.optimize.minimize(
                fun,
                V.flatten(),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-5},
            )
            V = res.x.reshape((self.n, self.d))

        # Enforce strict spherical constraint
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        return V / norms

    def perturb(self, points: np.ndarray, intensity: float, seed=None) -> np.ndarray:
        """
        Orchestrates surgical, structure-preserving modifications across multiple scales
        to navigate the complex energy landscape and free locked sub-structures.
        """
        # Re-use RNG properly to maintain trajectory unless actively overridden
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        new_points = points.copy()

        # -----------------------------------------------------------
        # 1. Active-Set Targeting (Micro-adjustments)
        # -----------------------------------------------------------
        ips = np.dot(new_points, new_points.T)
        np.fill_diagonal(ips, -np.inf)
        max_ip = np.max(ips)

        # Target the bottleneck active pairs
        threshold = max_ip - 0.03 * (1.0 + intensity)
        active_pairs = np.argwhere(ips > threshold)

        for i, j in active_pairs:
            if i > j:
                continue  # Process each interaction pair once

            # Force localized repulsion
            direction = new_points[i] - new_points[j]
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-8:
                direction /= norm_dir
                step_size = rng.uniform(0.01, 0.05) * (intensity + 0.1)
                new_points[i] += step_size * direction
                new_points[j] -= step_size * direction

            # Inject localized orthogonal tangent space noise to break symmetry
            for idx in (i, j):
                noise = rng.standard_normal(self.d)
                noise -= np.dot(noise, new_points[idx]) * new_points[idx]
                new_points[idx] += rng.uniform(0.005, 0.02) * (intensity + 0.1) * noise

        # Add ambient topological vibration
        global_noise = rng.standard_normal((self.n, self.d))
        global_noise -= (
            np.sum(global_noise * new_points, axis=1, keepdims=True) * new_points
        )
        new_points += global_noise * 0.002 * (intensity + 0.1)

        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)

        # -----------------------------------------------------------
        # 2. Surrogate Relaxation (Meso-Scale Optimization)
        # -----------------------------------------------------------
        # Smaller s is more global, higher s is more strictly localized
        s_val = 2.0 + 10.0 * (1.0 - intensity)

        def fun_s(v_flat):
            v_reshaped = v_flat.reshape((self.n, self.d))
            val, grad = self.val_grad_s_energy(v_reshaped, float(s_val))
            return float(val), np.array(grad, dtype=np.float64).flatten()

        max_steps = int(5 + 15 * intensity)
        res = scipy.optimize.minimize(
            fun_s,
            new_points.flatten(),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_steps},
        )
        new_points = res.x.reshape((self.n, self.d))
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)

        # -----------------------------------------------------------
        # 3. Manifold / Subspace Moves (Macro Topological Rotations)
        # -----------------------------------------------------------
        if intensity > 0.3:
            num_to_rotate = int(self.n * rng.uniform(0.1, 0.5 * intensity))
            if num_to_rotate > 0:
                # Select a contiguous "cap" of points on the hypersphere
                center_idx = rng.integers(0, self.n)
                center = new_points[center_idx]
                dists = np.linalg.norm(new_points - center, axis=1)
                subset_indices = np.argsort(dists)[:num_to_rotate]

                # Generate an SO(d) rotation via Lie Algebra exponential
                U_random = rng.standard_normal((self.d, self.d))
                Q, _ = np.linalg.qr(U_random)
                A = Q - Q.T  # Skew-symmetric
                A /= np.linalg.norm(A) + 1e-8
                theta = rng.uniform(0.1, 0.5) * intensity

                R = scipy.linalg.expm(theta * A)
                # Rotate the subset, introducing shifting fault lines in the code
                new_points[subset_indices] = np.dot(new_points[subset_indices], R.T)

        # Final strict spherical constraint projection
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        return new_points


def entrypoint():
    return Improver
