r"""
\section*{Algorithm Explanation}

The problem of finding a set of points on $S^{d-1}$ that minimizes the maximum pairwise cosine similarity is equivalent to the generalized Tammes problem (or minimizing the infinite-Riesz energy).

\subsection*{Optimization Method (\texttt{improve})}
The \texttt{improve} method employs a continuous, differentiable relaxation of the min-max objective using the LogSumExp (LSE) function:
$$ \mu_\alpha(X) = M + \frac{1}{\alpha} \log \sum_{i < j} \exp(\alpha (x_i \cdot x_j - M)) $$
where $M = \max_{i < j} (x_i \cdot x_j)$ is treated as a computational constant in the backward pass (via \texttt{jax.lax.stop\_gradient}) to prevent gradient singularities and numerical overflow.
We perform Riemannian-like optimization by extending the objective to $\mathbb{R}^{n \times d}$ and mapping to the sphere, coupled with a quadratic penalty on the norms to ensure full-rank Hessians for the L-BFGS-B optimizer:
$$ \mathcal{L}_\alpha(Y) = \mu_\alpha\left(\frac{Y}{\|Y\|}\right) + \lambda \frac{1}{n} \sum_{i=1}^n (\|y_i\| - 1)^2 $$
We optimize this surrogate using L-BFGS-B via \texttt{JAX} for exact, highly-optimized gradients. The temperature parameter $\alpha$ is geometrically annealed ($\alpha \in \{20, 50, 100, 300, 800\}$) to sequentially track the solution path from a smooth global landscape to the exact minimax configuration.

\subsection*{Perturbation Orchestration (\texttt{perturb})}
To escape treacherous local minima and saddle points inherent to dense sphere packings, \texttt{perturb} orchestrates a portfolio of stochastic, topological, and gradient-based heuristics:
\begin{itemize}
    \item \textbf{Active-Set Targeting:} Isolates the subset of points participating in the maximum inner products (the "bottleneck" active set). These specific points are subjected to localized geometric repulsions from their nearest neighbors and stochastic tangential shifts.
    \item \textbf{Surrogate Relaxation:} Temporarily morphs the loss landscape into a generalized Riesz $s$-energy ($E_s = \sum_{i<j} \|x_i - x_j\|^{-s}$) and runs a few steps of L-BFGS-B. This dissolves highly crystalline local minima and smoothly drifts the points into a neighboring attraction basin.
    \item \textbf{Manifold \& Subspace Twists:} Global topological operations apply rotations within randomly sampled 2D hyperplanes, scaling the rotation angle by a point's projection on a tertiary axis, effectively twisting the configuration along the manifold.
    \item \textbf{Active Reflections:} Reflects a subset of active points across a random hyperplane to cleanly break geometric invariants and jump to a topologically distinct configuration.
\end{itemize}
The orchestrator dynamically samples from this portfolio, using the \texttt{intensity} scalar to modulate step sizes, active-set fractions, and optimization bounds.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "True"

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


class Improver:
    def __init__(self, n: int, d: int, seed: int = 0):
        """
        Args:
            n: Number of points.
            d: Dimension of the space (points lie on S^{d-1} in R^d).
            seed: Random seed.
        """
        self.n = n
        self.d = d
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Precompute strict upper triangular indices to avoid diagonal and double counting
        i_idx, j_idx = jnp.triu_indices(n, k=1)

        # 1. Primary Min-Max Objective (LogSumExp)
        def loss_fn(Y_flat, alpha):
            Y = Y_flat.reshape((n, d))
            norms = jnp.linalg.norm(Y, axis=1, keepdims=True)
            X = Y / (norms + 1e-12)

            C = jnp.dot(X, X.T)
            C_ij = C[i_idx, j_idx]

            # Use stop_gradient to prevent JAX from unnecessarily trying to branch through max
            max_C = jax.lax.stop_gradient(jnp.max(C_ij))
            lse = max_C + jnp.log(jnp.sum(jnp.exp(alpha * (C_ij - max_C)))) / alpha

            norm_penalty = 1.0 * jnp.mean((norms.flatten() - 1.0) ** 2)
            return lse + norm_penalty

        self.val_and_grad = jax.jit(jax.value_and_grad(loss_fn, argnums=0))

        # 2. Surrogate Relaxation Objective (Riesz s-energy)
        def surrogate_loss_fn(Y_flat, s):
            Y = Y_flat.reshape((n, d))
            norms = jnp.linalg.norm(Y, axis=1, keepdims=True)
            X = Y / (norms + 1e-12)

            C = jnp.dot(X, X.T)
            C_ij = C[i_idx, j_idx]

            dist_sq = jnp.maximum(2.0 - 2.0 * C_ij, 1e-4)
            energy = jnp.sum(dist_sq ** (-s / 2.0))

            norm_penalty = 1.0 * jnp.mean((norms.flatten() - 1.0) ** 2)
            return energy + norm_penalty

        self.surrogate_val_and_grad = jax.jit(
            jax.value_and_grad(surrogate_loss_fn, argnums=0)
        )

    def improve(self, points: np.ndarray, seed=None) -> np.ndarray:
        """
        Refines an existing configuration to minimize the maximum inner product.
        Strictly enforces the spherical constraint (||x|| = 1) in the output.
        """
        if seed is not None:
            np.random.seed(seed)

        # Annealing schedule to bridge from a smooth global landscape to the exact min-max problem
        alphas = [20.0, 50.0, 100.0, 300.0, 800.0]
        Y_flat = points.flatten()

        def wrap_val_and_grad(x, a):
            val, grad = self.val_and_grad(x, a)
            return float(val), np.array(grad, dtype=np.float64)

        for alpha in alphas:
            res = minimize(
                fun=wrap_val_and_grad,
                x0=Y_flat,
                args=(alpha,),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 150, "ftol": 1e-7, "gtol": 1e-5},
            )
            Y_flat = res.x

        # Hard projection strictly onto the hypersphere
        Y = Y_flat.reshape((self.n, self.d))
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        return Y / norms

    def perturb(self, points: np.ndarray, intensity: float, seed=None) -> np.ndarray:
        """
        Applies a highly non-trivial orchestration of surgical structural or
        stochastic modifications to escape local minima while preserving essence.
        """
        rng = np.random.default_rng(
            seed if seed is not None else self.rng.integers(1 << 31)
        )
        new_points = points.copy()

        # Portfolio of orchestrated strategies dynamically chosen
        strategy = rng.choice(
            ["active_set", "surrogate", "twist", "reflection"], p=[0.4, 0.3, 0.15, 0.15]
        )

        if strategy == "active_set":
            # 1. Target the "bottleneck" pairs causing the worst cosine similarity
            C = np.dot(points, points.T)
            np.fill_diagonal(C, -np.inf)
            max_c = np.max(C, axis=1)

            frac = 0.05 + 0.15 * intensity
            k = max(2, int(self.n * frac))
            active_indices = np.argsort(max_c)[-k:]

            for idx in active_indices:
                enemy_idx = np.argmax(C[idx])
                enemy = points[enemy_idx]

                # Derive localized repelling vector
                repel_dir = points[idx] - enemy
                repel_dir -= np.dot(repel_dir, points[idx]) * points[idx]
                nrm = np.linalg.norm(repel_dir)

                if nrm > 1e-8:
                    repel_dir /= nrm
                else:
                    # Resolve exact overlaps with tangential random displacement
                    repel_dir = rng.normal(size=self.d)
                    repel_dir -= np.dot(repel_dir, points[idx]) * points[idx]
                    repel_dir /= np.linalg.norm(repel_dir) + 1e-12

                noise = rng.normal(size=self.d)
                noise -= np.dot(noise, points[idx]) * points[idx]
                noise -= np.dot(noise, repel_dir) * repel_dir
                n_nrm = np.linalg.norm(noise)
                if n_nrm > 1e-8:
                    noise /= n_nrm
                else:
                    noise = np.zeros_like(noise)

                move_dir = repel_dir * 0.7 + noise * 0.3
                move_nrm = np.linalg.norm(move_dir)
                if move_nrm > 1e-8:
                    move_dir /= move_nrm
                else:
                    move_dir = repel_dir

                # Apply localized geodesic step
                step = intensity * rng.uniform(0.05, 0.25)
                new_pt = np.cos(step) * points[idx] + np.sin(step) * move_dir
                new_points[idx] = new_pt / np.linalg.norm(new_pt)

        elif strategy == "surrogate":
            # 2. Relax objective into Riesz Energy and slide into deeper, alternative basins
            s = float(rng.choice([1.0, 2.0, 3.0]))

            def wrap_surrogate(x, s_val):
                val, grad = self.surrogate_val_and_grad(x, s_val)
                return float(val), np.array(grad, dtype=np.float64)

            res = minimize(
                fun=wrap_surrogate,
                x0=points.flatten(),
                args=(s,),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": int(10 + 30 * intensity)},
            )
            new_points = res.x.reshape((self.n, self.d))

        elif strategy == "twist":
            # 3. Manifold Twists: Globally twist points inside a randomly sampled plane
            u = rng.normal(size=self.d)
            u /= np.linalg.norm(u)
            v = rng.normal(size=self.d)
            v -= np.dot(v, u) * u
            v /= np.linalg.norm(v)

            w = rng.normal(size=self.d)
            w /= np.linalg.norm(w)

            base_angle = intensity * rng.uniform(0.1, 0.4)
            angles = base_angle * np.dot(
                points, w
            )  # Rotation scales continuously via 3rd dimension axis
            cos_a = np.cos(angles)[:, None]
            sin_a = np.sin(angles)[:, None]

            proj_u = np.dot(points, u)[:, None]
            proj_v = np.dot(points, v)[:, None]

            new_points -= proj_u * u + proj_v * v
            new_points += (proj_u * cos_a - proj_v * sin_a) * u
            new_points += (proj_u * sin_a + proj_v * cos_a) * v

        elif strategy == "reflection":
            # 4. Invert points across a local active set to cleanly break problematic geometric invariants
            C = np.dot(points, points.T)
            np.fill_diagonal(C, -np.inf)
            max_c = np.max(C, axis=1)

            k = max(2, int(self.n * intensity * 0.2))
            active_indices = np.argsort(max_c)[-k:]

            normal = rng.normal(size=self.d)
            normal /= np.linalg.norm(normal)

            for idx in active_indices:
                p = points[idx]
                proj = np.dot(p, normal)
                new_pt = (
                    p - 2.0 * proj * normal
                )  # Householder reflection maintains unit norm natively
                new_points[idx] = new_pt / np.linalg.norm(new_pt)

        # Reproject to guarantee constraints
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        return new_points / norms

    def generate_config(self, seed=None) -> np.ndarray:
        """
        Generates a valid initial configuration of n points uniformly mapped onto the unit sphere.
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        pts = rng.normal(size=(self.n, self.d))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        return pts / norms


def entrypoint():
    return Improver
