import collections

import jax
from jax import jit, value_and_grad, vmap
import jax.numpy as jnp
import numpy as np
import scipy.optimize

# Enable 64-bit precision for high geometric fidelity
jax.config.update("jax_enable_x64", True)

"""
HEXAGON PACKING IMPROVER
========================

ALGORITHM OVERVIEW:
This class solves the packing problem using a differentiable physics-based approach.

1.  **Geometric Representation**:
    -   **Unit Hexagons**: Defined by circumradius $R=1$ (Side=1). Vertices are computed dynamically based on $(x,y)$ and $\theta$.
    -   **Enclosing Hexagon**: Regular flat-topped hexagon. Boundary defined by 3 pairs of parallel lines (normals at $30^\circ, 90^\circ, 150^\circ$).

2.  **Constraint Logic**:
    -   **Non-Overlap**: Implemented using the **Separating Axis Theorem (SAT)**.
        -   For every pair, we project vertices onto 6 axes (3 from each hex).
        -   We calculate separation $d$. Overlap occurs if $d < 0$.
        -   **Safety Margin**: We enforce $d \ge \epsilon$ (e.g., 0.001) by penalizing $\text{ReLU}(-(d - \epsilon))^2$. This ensures that even with slight numerical imprecision, the physical result remains non-overlapping.
    -   **Containment**: Projects all vertices onto the container's face normals. Violations are penalized quadratically.

3.  **Optimization Pipeline**:
    -   **Gradient Descent**: Uses `scipy.optimize.minimize` (L-BFGS-B) with JAX-computed gradients.
    -   **Iterative Refinement**:
        1.  **Solve**: Minimize side length $L$ while respecting constraints.
        2.  **Verify**: Check for hard constraint violations.
        3.  **Repair Loop**: If overlaps exist (due to local minima):
            -   Slightly expand the container $L$.
            -   Apply a random "kick" (perturbation) to unit hexagons to break symmetrical jams.
            -   Re-optimize with $L$ fixed and extreme penalty weights to force separation.
            -   Repeat until valid or timeout.
"""


class Improver:
    def __init__(self, hex_num=11, seed: int = 0):
        self.hex_num = hex_num
        self.seed = seed
        self._init_jit_funcs()

    def _init_jit_funcs(self):
        """Compiles the JAX geometry and loss functions."""
        N = self.hex_num
        SQRT3 = np.sqrt(3.0)

        # Enclosing Hexagon Normals (30, 90, 150 deg)
        # Used for containment check of flat-topped hexagon
        env_normals = jnp.array([[SQRT3 / 2.0, 0.5], [0.0, 1.0], [-SQRT3 / 2.0, 0.5]])

        # --- GEOMETRY KERNELS ---

        def get_verts(c, theta):
            """Returns (6, 2) vertices of a unit hexagon (R=1)."""
            # Vertices at 0, 60, ... 300 degrees + theta
            angles = (
                jnp.array(
                    [
                        0.0,
                        jnp.pi / 3,
                        2 * jnp.pi / 3,
                        jnp.pi,
                        4 * jnp.pi / 3,
                        5 * jnp.pi / 3,
                    ]
                )
                + theta
            )
            vx = c[0] + jnp.cos(angles)
            vy = c[1] + jnp.sin(angles)
            return jnp.stack([vx, vy], axis=1)

        def get_axes(theta):
            """Returns (3, 2) face normals for SAT checks."""
            # Normals at 30, 90, 150 degrees + theta
            angles = jnp.array([jnp.pi / 6, jnp.pi / 2, 5 * jnp.pi / 6]) + theta
            return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

        # --- LOSS FUNCTIONS ---

        def containment_penalty(centers, angles, L):
            """Penalty for protruding outside the enclosing hexagon."""
            # Distance from center to flat edge = L * sqrt(3)/2
            H = L * SQRT3 / 2.0

            def check_single(c, a):
                verts = get_verts(c, a)
                # Project vertices onto container normals
                projs = jnp.dot(env_normals, verts.T)
                # Violation if |projection| > H
                return jnp.sum(jnp.square(jnp.maximum(jnp.abs(projs) - H, 0.0)))

            return jnp.sum(vmap(check_single)(centers, angles))

        def pair_overlap_penalty(c1, a1, c2, a2, margin):
            """
            SAT Overlap Check with Safety Margin.
            Returns squared penalty if separation < margin.
            """
            # Gather 6 axes
            ax = jnp.concatenate([get_axes(a1), get_axes(a2)], axis=0)

            # Project vertices
            v1 = get_verts(c1, a1)
            v2 = get_verts(c2, a2)
            p1 = jnp.dot(ax, v1.T)
            p2 = jnp.dot(ax, v2.T)

            # Bounds
            min1, max1 = jnp.min(p1, axis=1), jnp.max(p1, axis=1)
            min2, max2 = jnp.min(p2, axis=1), jnp.max(p2, axis=1)

            # Separation d
            # If d > 0, separated. We want d >= margin.
            d = jnp.maximum(min1 - max2, min2 - max1)

            # "Least separation" across all axes
            sep = jnp.max(d)

            # Violation if sep < margin
            # Penalty = (margin - sep)^2
            return jnp.square(jnp.maximum(margin - sep, 0.0))

        def total_overlap_penalty(centers, angles, margin):
            """Sum of overlaps for all unique pairs."""
            i_idx, j_idx = jnp.triu_indices(N, k=1)

            c1 = centers[i_idx]
            a1 = angles[i_idx]
            c2 = centers[j_idx]
            a2 = angles[j_idx]

            penalties = vmap(pair_overlap_penalty, in_axes=(0, 0, 0, 0, None))(
                c1, a1, c2, a2, margin
            )
            return jnp.sum(penalties)

        # --- OBJECTIVE ---

        def objective(params, w_ov, w_cn, fix_L, target_L, margin):
            """
            params: [L, c_x... c_y..., theta...]
            """
            L_val = params[0]
            L = jax.lax.select(fix_L, target_L, L_val)

            c_flat = params[1 : 1 + 2 * N]
            centers = c_flat.reshape((N, 2))
            angles = params[1 + 2 * N :]

            loss_ov = total_overlap_penalty(centers, angles, margin)
            loss_cn = containment_penalty(centers, angles, L)

            obj = (0.0 if fix_L else L) + w_ov * loss_ov + w_cn * loss_cn
            return obj

        # Compile
        self.jit_val_grad = jit(
            value_and_grad(objective, argnums=0), static_argnames=["fix_L"]
        )
        self.calc_overlap = jit(total_overlap_penalty)
        self.calc_contain = jit(containment_penalty)

    def generate_config(self, seed=None) -> tuple[np.ndarray, np.ndarray]:
        """Generates a hexagonal lattice packing."""
        if seed is not None:
            np.random.seed(seed)

        N = self.hex_num
        spacing = 2.1  # Safe spacing

        points = []
        queue = collections.deque([(0, 0)])
        visited = {(0, 0)}

        # Spiral generation
        while len(points) < N:
            q, r = queue.popleft()
            cx = spacing * (q + 0.5 * r)
            cy = spacing * (np.sqrt(3) / 2 * r)
            points.append([cx, cy])

            for dq, dr in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in visited:
                    visited.add((nq, nr))
                    queue.append((nq, nr))

        centers = np.array(points[:N])
        centers -= np.mean(centers, axis=0)  # Center at origin
        angles = np.zeros(N)
        return centers, angles

    def perturb(self, input_config, intensity, seed=None):
        if seed is not None:
            np.random.seed(seed)
        c, a = input_config
        c = c.copy()
        a = a.copy()

        # Noise
        c += np.random.normal(0, 0.3 * intensity, c.shape)
        a += np.random.normal(0, 1.5 * intensity, a.shape)

        # Swaps
        if intensity > 0.4 and self.hex_num > 1:
            idx = np.random.choice(self.hex_num, 2, replace=False)
            c[idx[0]], c[idx[1]] = c[idx[1]], c[idx[0]]

        return c, a

    def improve(self, input_config, seed=None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            np.random.seed(seed)

        centers, angles = input_config
        N = self.hex_num

        # Determine initial bounding box
        dists = np.linalg.norm(centers, axis=1)
        L_init = np.max(dists) + 1.5

        params = np.concatenate([[L_init], centers.ravel(), angles.ravel()])

        # SAFETY MARGIN: We enforce separation of 0.005 during optimization
        # to guarantee the final result passes the strict 0.0 check.
        opt_margin = 0.005

        def optimize_stage(p0, w_ov, w_cn, fix_L=False, tgt_L=0.0, iters=200):
            def func(p):
                val, g = self.jit_val_grad(p, w_ov, w_cn, fix_L, tgt_L, opt_margin)
                return float(val), np.array(g, dtype=np.float64)

            bounds = [(1.0, None)] + [(None, None)] * (3 * N)

            res = scipy.optimize.minimize(
                func,
                p0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": iters, "ftol": 1e-9},
            )
            return res.x

        # 1. Main Optimization (Minimize L)
        # Increasing weights schedule
        params = optimize_stage(params, w_ov=10.0, w_cn=10.0, iters=100)
        params = optimize_stage(params, w_ov=1000.0, w_cn=1000.0, iters=200)
        params = optimize_stage(params, w_ov=1e5, w_cn=1e5, iters=300)

        # 2. Validation & Repair
        # We verify with margin=0.0 (strict check)
        def get_status(p_vec):
            L = p_vec[0]
            c = p_vec[1 : 1 + 2 * N].reshape((N, 2))
            a = p_vec[1 + 2 * N :]
            ov = float(self.calc_overlap(c, a, 0.0))  # Strict
            cn = float(self.calc_contain(c, a, L))
            return ov, cn, L, c, a

        ov, cn, L_curr, c_curr, a_curr = get_status(params)

        # Repair Loop
        repair_attempts = 0
        while (ov > 1e-6 or cn > 1e-6) and repair_attempts < 10:
            repair_attempts += 1

            # Expand L slightly to alleviate pressure
            L_target = L_curr * (1.01 + 0.01 * repair_attempts)

            # PERTURBATION: If we are stuck in overlap, we must kick the system.
            # We apply noise directly to the parameters.
            c_perturb, a_perturb = self.perturb(
                (c_curr, a_curr), intensity=0.1 * repair_attempts
            )

            p_repair = np.concatenate(
                [[L_target], c_perturb.ravel(), a_perturb.ravel()]
            )

            # Re-optimize with Fixed L and Massive Weights
            # Use a slightly smaller margin in repair to just get valid,
            # or keep it high to be safe. Keeping it 0.005 is safe.
            params = optimize_stage(
                p_repair, w_ov=1e8, w_cn=1e8, fix_L=True, tgt_L=L_target, iters=300
            )

            # Update L in params (since fix_L ignores it)
            params[0] = L_target

            ov, cn, L_curr, c_curr, a_curr = get_status(params)

        # Final Formatting
        centers_out = params[1 : 1 + 2 * N].reshape((N, 2))
        angles_out = params[1 + 2 * N :]
        angles_out = np.mod(angles_out, 2 * np.pi)

        return centers_out, angles_out


def entrypoint():
    return Improver
