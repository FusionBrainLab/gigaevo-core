"""
Spherical Code Optimizer

This module implements a state-of-the-art solver for generating and optimizing
spherical codes, which are configurations of $n$ points on the unit hypersphere
$S^{d-1} \subset \mathbb{R}^d$ that minimize the maximum pairwise inner product:
$$ \mu(X) = \max_{1 \leq i < j \leq n} (x_i \cdot x_j) $$

### Optimization Strategy (`improve`)
The non-smooth minimax objective is notoriously difficult to optimize directly.
Instead, we use a smooth surrogate relaxation via the LogSumExp function:
$$ J_\alpha(X) = \frac{1}{\alpha} \log \left( \sum_{i < j} \exp(\alpha (x_i \cdot x_j)) \right) $$
As $\alpha \to \infty$, $J_\alpha(X) \to \mu(X)$.
We apply an annealing sequence of increasing $\alpha \in \{40, 100, 200, 400\}$
and optimize $J_\alpha$ using the L-BFGS-B algorithm. To strictly enforce the
spherical constraint without complex Riemannian retractions, we over-parameterize
the points as $Y \in \mathbb{R}^{n \times d}$ and project them onto the sphere
during objective evaluation: $x_i = Y_i / \|Y_i\|_2$. The gradient naturally
ignores radial components, ensuring robust convergence on the manifold.

### Perturbation Heuristic (`perturb`)
To escape shallow local minima, we implement a highly surgical "Hole Digging"
active-set perturbation:
1. **Bottleneck Identification:** We construct the pairwise similarity matrix
   and greedily identify the $k$ most problematic points (those realizing the
   highest inner products and thus forming structural bottlenecks).
2. **Hole Search:** We uniformly sample $M = 500,000$ candidate points on $S^{d-1}$
   and evaluate their maximum similarity to the non-removed points. We greedily
   select the $k$ candidates with the lowest maximum similarity (the deepest "holes").
3. **Surgical Replacement:** The bottleneck points are directly teleported into
   these deep structural holes.
This breaks local symmetries and optimally redistributes congestion without
destroying the well-packed global configuration.
"""

import jax
from jax import jit, value_and_grad
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

# Ensure JAX uses 64-bit precision to maintain numerical stability
# during high-alpha LogSumExp and accurate L-BFGS convergence.
jax.config.update("jax_enable_x64", True)


@jit
def loss_fn(Y, alpha):
    """
    Computes the LogSumExp surrogate objective for the maximum inner product.
    Strictly projects over-parameterized Y onto the unit hypersphere.
    """
    norms = jnp.linalg.norm(Y, axis=1, keepdims=True)
    X = Y / jnp.maximum(norms, 1e-12)

    # Compute pairwise cosine similarities
    dots = jnp.dot(X, X.T)

    # Mask out diagonal (self-similarity) and lower triangle to avoid double counting
    mask = jnp.triu(jnp.ones_like(dots), k=1)
    dots_masked = jnp.where(mask == 1, dots, -jnp.inf)

    # Smooth max approximation
    return jax.nn.logsumexp(alpha * dots_masked) / alpha


# Compile the value and gradient function once to minimize JIT overhead.
val_and_grad_fn = jit(value_and_grad(loss_fn))


def scipy_obj_grad(Y_flat, n, d, alpha):
    """Wrapper function to interface JAX with SciPy's L-BFGS-B optimizer."""
    Y = Y_flat.reshape((n, d))
    v, g = val_and_grad_fn(Y, alpha)
    return float(v), np.array(g, dtype=np.float64).flatten()


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
        np.random.seed(seed)

    def generate_config(self, seed=None) -> np.ndarray:
        """
        Generates a valid initial configuration of n points on the unit sphere.
        """
        if seed is not None:
            np.random.seed(seed)

        points = np.random.normal(size=(self.n, self.d))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / np.maximum(norms, 1e-12)

    def improve(self, points: np.ndarray, seed=None) -> np.ndarray:
        """
        Refines an existing configuration to minimize the maximum inner product.
        Must strictly enforce the spherical constraint (||x|| = 1) in the output.
        """
        if seed is not None:
            np.random.seed(seed)

        Y_flat = points.flatten()
        n, d = self.n, self.d

        # Progressive relaxation sequence of the surrogate sharpness
        alphas = [40.0, 100.0, 200.0, 400.0]

        for alpha in alphas:
            res = minimize(
                fun=scipy_obj_grad,
                x0=Y_flat,
                args=(n, d, alpha),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 250, "ftol": 1e-6, "gtol": 1e-5},
            )
            Y_flat = res.x

        # Final strict projection
        Y = Y_flat.reshape((n, d))
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        X = Y / np.maximum(norms, 1e-12)
        return X

    def perturb(self, points: np.ndarray, intensity: float, seed=None) -> np.ndarray:
        """
        Applies a 'Hole Digging' active-set perturbation. Identifies strictly
        bottleneck pairs and surgically teleports them into the deepest structurally
        available holes on the hypersphere.
        """
        if seed is not None:
            np.random.seed(seed)

        if intensity <= 0.0:
            return points.copy()

        n, d = points.shape
        # Scale perturbation target size dynamically based on intensity limit
        k = max(1, int(intensity * 0.1 * n))
        k = min(k, n - 1)

        # 1. Identify the 'k' absolute worst structural bottleneck points
        D = points @ points.T
        np.fill_diagonal(D, -np.inf)
        D_copy = D.copy()

        removed_indices = set()
        for _ in range(k):
            # Find worst pair
            idx = np.argmax(D_copy)
            i, j = np.unravel_index(idx, D_copy.shape)

            # Select the one with worse overall spatial crowding using the second maximum inner product
            D_i = D[i].copy()
            D_i[j], D_i[i] = -np.inf, -np.inf
            second_max_i = np.max(D_i)

            D_j = D[j].copy()
            D_j[i], D_j[j] = -np.inf, -np.inf
            second_max_j = np.max(D_j)

            remove_idx = int(i if second_max_i > second_max_j else j)

            # Fallback (mathematically unreachable due to subsequent masking, but defensively coded)
            if remove_idx in removed_indices:
                remove_idx = int(j if remove_idx == i else i)

            removed_indices.add(remove_idx)

            # Nullify its influence in the matrix to find purely distinct future bottlenecks
            D_copy[remove_idx, :] = -np.inf
            D_copy[:, remove_idx] = -np.inf

        removed_indices_list = list(removed_indices)

        # 2. Extract retained well-packed structural subset
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[removed_indices_list] = False
        X_rem = points[keep_mask]

        # 3. Large-scale generative Hole Search
        M = 500000
        C = np.random.normal(size=(M, d))
        norms = np.linalg.norm(C, axis=1, keepdims=True)
        C = C / np.maximum(norms, 1e-12)

        # Efficiently compute spatial gap depth
        v_max = np.full(M, -np.inf)
        batch_size = 50000
        for b in range(0, M, batch_size):
            C_batch = C[b : b + batch_size]
            v_max[b : b + batch_size] = np.max(C_batch @ X_rem.T, axis=1)

        final_points = points.copy()

        # 4. Surgically place bottlenecks into holes iteratively avoiding mutual hole clustering
        for i in range(len(removed_indices_list)):
            best_idx = np.argmin(v_max)
            best_c = C[best_idx]
            final_points[removed_indices_list[i]] = best_c

            if i < len(removed_indices_list) - 1:
                # Modulate the landscape evaluating the new member's footprint
                sim = C @ best_c
                v_max = np.maximum(v_max, sim)

        return final_points


def entrypoint():
    return Improver
