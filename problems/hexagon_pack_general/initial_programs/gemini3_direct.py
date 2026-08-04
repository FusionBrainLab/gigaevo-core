import jax
from jax import jit, value_and_grad, vmap
import jax.numpy as jnp
import numpy as np
import scipy.optimize

# Ensure 64-bit precision for geometric robustness
jax.config.update("jax_enable_x64", True)

# ==========================================
# GEOMETRIC KERNELS (JAX)
# ==========================================


@jit
def get_hex_vertices(center, theta, scale):
    """
    Computes vertices of a regular hexagon with circumradius `scale` (side length).
    Flat-topped orientation relative to local frame.
    """
    # Angles for flat-topped hex: 0, 60, 120, 180, 240, 300 degrees
    # theta adds rotation.
    angles = theta + jnp.array(
        [0.0, jnp.pi / 3, 2 * jnp.pi / 3, jnp.pi, 4 * jnp.pi / 3, 5 * jnp.pi / 3]
    )
    vx = scale * jnp.cos(angles)
    vy = scale * jnp.sin(angles)
    # Shape: (6, 2)
    return center + jnp.stack([vx, vy], axis=1)


@jit
def get_normals(theta):
    """
    Returns the 3 unique unit normal vectors for a regular hexagon.
    Normals are perpendicular to the edges (at 30, 90, 150 deg relative to vertices).
    """
    angles = theta + jnp.array([jnp.pi / 6, jnp.pi / 2, 5 * jnp.pi / 6])
    return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)


@jit
def overlap_amount(c1, t1, c2, t2, margin):
    """
    Calculates intersection depth using Separating Axis Theorem (SAT).
    Includes a `margin` to enforce strict separation (simulates larger radius).

    Returns:
        0.0 if separated by at least `margin`
        positive value if overlapping or closer than `margin`
    """
    # Effective radius including safety buffer
    r = 1.0 + margin

    # 1. Candidate Axes (Normals from both polygons)
    n1 = get_normals(t1)
    n2 = get_normals(t2)
    axes = jnp.concatenate([n1, n2])  # Shape (6, 2)

    # 2. Vertices (inflated by r)
    v1 = get_hex_vertices(c1, t1, r)
    v2 = get_hex_vertices(c2, t2, r)

    # 3. Project vertices onto axes
    dots1 = jnp.dot(v1, axes.T)  # (6, 6)
    dots2 = jnp.dot(v2, axes.T)

    min1 = jnp.min(dots1, axis=0)
    max1 = jnp.max(dots1, axis=0)
    min2 = jnp.min(dots2, axis=0)
    max2 = jnp.max(dots2, axis=0)

    # 4. Check overlap on each axis
    # Overlap = intersection length of intervals [min1, max1] and [min2, max2]
    # If separated, this value is negative.
    overlaps = jnp.minimum(max1, max2) - jnp.maximum(min1, min2)

    # 5. SAT: Polygons intersect if and only if they overlap on ALL axes.
    # The penetration depth is the MINIMUM overlap among the axes.
    penetration = jnp.min(overlaps)

    # 6. ReLU: We only care if penetration > 0
    return jax.nn.relu(penetration)


@jit
def compute_total_overlap(centers, angles, margin):
    """
    Calculates sum of pairwise overlaps using full vectorization.
    """
    N = centers.shape[0]

    # Define pairwise operation
    def pairwise(c1, t1, c2, t2):
        return overlap_amount(c1, t1, c2, t2, margin)

    # Map over all pairs (N, N)
    # vmap over rows (i), then columns (j)
    matrix = vmap(
        lambda c1, t1: vmap(lambda c2, t2: pairwise(c1, t1, c2, t2))(centers, angles)
    )(centers, angles)

    # Mask to keep unique pairs (i < j)
    # triu with k=1 excludes diagonal
    mask = jnp.triu(jnp.ones((N, N)), k=1)

    return jnp.sum(matrix * mask)


@jit
def compute_enclosing_L(centers, angles):
    """
    Calculates the circumradius L of the minimal FLAT-TOPPED enclosing hexagon.
    """
    # Vertices of actual unit hexagons (scale=1.0)
    verts = vmap(lambda c, t: get_hex_vertices(c, t, 1.0))(centers, angles)
    all_points = verts.reshape((-1, 2))

    x = jnp.abs(all_points[:, 0])
    y = jnp.abs(all_points[:, 1])

    # Flat-topped hexagon metric
    # A point (x,y) is inside if distance d(x,y) <= Apothem
    # d(x,y) = max(|y|, (|y| + sqrt(3)|x|)/2)
    req_apothem = jnp.maximum(y, (y + jnp.sqrt(3) * x) / 2.0)

    max_apothem = jnp.max(req_apothem)

    # Relation: Apothem = L * sqrt(3)/2
    # L = Apothem * 2/sqrt(3)
    L = max_apothem * 2.0 / jnp.sqrt(3)
    return L


# ==========================================
# IMPROVER CLASS
# ==========================================


class Improver:
    def __init__(self, hex_num=11, seed: int = 0):
        self.hex_num = hex_num
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Compile JAX functions for this N
        self._compile_functions()

    def _compile_functions(self):
        N = self.hex_num

        def loss_fn(params_flat, w_ov, w_L, margin):
            # Unpack
            centers = params_flat[: 2 * N].reshape((N, 2))
            angles = params_flat[2 * N :]

            # Components
            L = compute_enclosing_L(centers, angles)
            ov = compute_total_overlap(centers, angles, margin)

            # Penalty: Squared overlap for smooth gradients near zero
            # Adding linear term ensures non-zero gradient if overlap is huge
            penalty = w_ov * (ov**2 + ov)

            return w_L * L + penalty

        self.loss_and_grad_fn = jit(value_and_grad(loss_fn))
        self.check_overlap_fn = jit(compute_total_overlap)
        self.calc_L_fn = jit(compute_enclosing_L)

    def improve(
        self, input_config: tuple[np.ndarray, np.ndarray], seed=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Refines the packing using constrained optimization with safety margins.
        """
        centers, angles = input_config
        N = self.hex_num

        # Ensure correct types
        centers = centers.astype(np.float64)
        angles = angles.astype(np.float64)

        x = np.concatenate([centers.flatten(), angles.flatten()])

        # Strategy:
        # Start with a small safety margin to force strict separation.
        # If successful, we have a valid config.
        margin = 1e-3  # Buffer radius
        w_ov = 100.0  # Initial penalty weight
        w_L = 1.0  # Objective weight

        best_valid_x = None
        min_L_valid = float("inf")

        # Iterative Hardening
        for stage in range(5):
            # Scipy Optimization
            def func(x_in):
                v, g = self.loss_and_grad_fn(jnp.array(x_in), w_ov, w_L, margin)
                return float(v), np.array(g, dtype=np.float64)

            res = scipy.optimize.minimize(
                func,
                x,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 2000, "ftol": 1e-8, "gtol": 1e-8},
            )
            x = res.x

            # Validation Check (with margin=0, actual constraint)
            c_curr = x[: 2 * N].reshape((N, 2))
            a_curr = x[2 * N :]

            ov_val = float(self.check_overlap_fn(c_curr, a_curr, 0.0))
            L_val = float(self.calc_L_fn(c_curr, a_curr))

            if ov_val < 1e-7:
                # Valid configuration found!
                if L_val < min_L_valid:
                    min_L_valid = L_val
                    best_valid_x = x.copy()

                # If we have a margin, we can try to reduce it to pack tighter
                if margin > 1e-5:
                    margin /= 2.0
                else:
                    # Margin is negligible, we are fully optimizing L
                    pass
            else:
                # Invalid: Increase penalty drastically
                w_ov *= 10.0

        # Return best valid result if found, else result of last iteration
        if best_valid_x is not None:
            x_final = best_valid_x
        else:
            x_final = x

        centers_final = x_final[: 2 * N].reshape((N, 2))
        angles_final = x_final[2 * N :]

        # Normalize angles
        angles_final = np.mod(angles_final, 2 * np.pi)

        return centers_final, angles_final

    def perturb(
        self, input_config: tuple[np.ndarray, np.ndarray], intensity: float, seed=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies structural changes to escape local optima.
        """
        centers, angles = input_config
        N = self.hex_num
        rng = np.random.default_rng(seed if seed is not None else self.seed)

        if intensity > 1.0:
            # High intensity: Reseed/Scramble
            # Randomize positions within a slightly expanded bounding box
            current_bound = np.max(np.abs(centers))
            limit = max(current_bound, 4.0)

            new_centers = rng.uniform(-limit, limit, size=(N, 2))
            new_angles = rng.uniform(0, 2 * np.pi, size=(N,))
            return new_centers, new_angles

        else:
            # Low intensity: Jiggle
            # Add Gaussian noise
            c_noise = rng.normal(0, 0.2 * intensity, size=(N, 2))
            a_noise = rng.normal(0, 0.4 * intensity, size=(N,))

            return centers + c_noise, np.mod(angles + a_noise, 2 * np.pi)

    def generate_config(self, seed=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a valid starting configuration using a spiral heuristic.
        """
        N = self.hex_num
        rng = np.random.default_rng(seed if seed is not None else self.seed)

        # Generate hexagonal grid points (axial q, r)
        queue = [(0, 0)]
        visited = {(0, 0)}
        coords = []
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

        while len(coords) < N:
            q, r = queue.pop(0)
            coords.append((q, r))

            # Randomize expansion direction
            iter_dirs = list(dirs)
            rng.shuffle(iter_dirs)
            for dq, dr in iter_dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in visited:
                    visited.add((nq, nr))
                    queue.append((nq, nr))

        # Convert to Flat-Topped Cartesian
        # Spacing factor > 1.0 to avoid initial overlaps
        spacing = 1.05
        centers = []
        for q, r in coords:
            # x = sqrt(3) * (q + r/2)
            # y = 1.5 * r
            cx = spacing * np.sqrt(3) * (q + r / 2.0)
            cy = spacing * 1.5 * r
            centers.append([cx, cy])

        centers = np.array(centers)
        centers -= np.mean(centers, axis=0)

        angles = rng.uniform(0, 2 * np.pi, size=(N,))

        return centers, angles


def entrypoint():
    return Improver


# =============================================================================
# DIRECT-INTERFACE ADAPTER (seed transferred from an ImprovEvolve initial).
# The Improver machinery above is unchanged. entrypoint() below replaces the
# class-returning entrypoint with the direct contract: one call spends its
# wall-clock budget on restarts of generate_config -> improve and returns the
# best configuration that passes the self-contained overlap check below.
# Falls back to a spaced row, which is valid for every hex_num by construction.
_DIRECT_IMPROVER_CLS = entrypoint()
_DIRECT_TIME_BUDGET_S = 240.0  # stay under the 300 s evaluation timeout
_DIRECT_MAX_RESTARTS = 64
_DIRECT_SAFETY_MARGIN = 1e-7


def _direct_vertices(centers, angles):
    offsets = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    theta = np.asarray(angles, dtype=float)[:, None] + offsets[None, :]
    return np.asarray(centers, dtype=float)[:, None, :] + np.stack(
        [np.cos(theta), np.sin(theta)], axis=-1
    )


def _direct_any_overlap(centers, angles):
    """Pairwise SAT on the 3+3 face normals; conservative by _DIRECT_SAFETY_MARGIN."""
    n = len(centers)
    verts = _direct_vertices(centers, angles)
    normal_offsets = np.array([np.pi / 6.0, np.pi / 2.0, 5.0 * np.pi / 6.0])
    theta = np.asarray(angles, dtype=float)[:, None] + normal_offsets[None, :]
    normals = np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # (n, 3, 2)
    centers = np.asarray(centers, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(centers[i] - centers[j])) >= 2.0:
                continue  # circumradius 1 each: distance >= 2 cannot overlap
            axes = np.concatenate([normals[i], normals[j]], axis=0)
            pi_ = verts[i] @ axes.T
            pj_ = verts[j] @ axes.T
            gap = np.maximum(
                pi_.min(axis=0) - pj_.max(axis=0), pj_.min(axis=0) - pi_.max(axis=0)
            )
            if gap.max() < _DIRECT_SAFETY_MARGIN:
                return True
    return False


def _direct_outer_side(centers, angles):
    verts = _direct_vertices(centers, angles).reshape(-1, 2)
    proj = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False) + np.pi / 6.0
    dirs = np.stack([np.cos(proj), np.sin(proj)], axis=-1)
    return 2.0 * float(np.max(np.abs(verts @ dirs.T))) / np.sqrt(3.0)


def _direct_fallback(hex_num):
    centers = np.stack(
        [2.0000001 * np.arange(hex_num, dtype=float), np.zeros(hex_num)], axis=-1
    )
    return centers, np.zeros(hex_num)


def _direct_clean(config, hex_num):
    centers = np.asarray(config[0], dtype=float)
    angles = np.asarray(config[1], dtype=float).reshape(-1)
    if centers.shape != (hex_num, 2) or angles.shape != (hex_num,):
        return None
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(angles))):
        return None
    return centers, np.mod(angles, 2.0 * np.pi)


def entrypoint(hex_num: int = 11, seed: int = 0):
    import time

    started = time.monotonic()
    improver = _DIRECT_IMPROVER_CLS(hex_num=hex_num, seed=seed)
    best = None
    best_side = float("inf")
    for attempt in range(_DIRECT_MAX_RESTARTS):
        restart_seed = int(seed) + 9973 * attempt
        try:
            config = improver.generate_config(seed=restart_seed)
            config = improver.improve(config, seed=restart_seed)
            cleaned = _direct_clean(config, hex_num)
            if cleaned is not None and not _direct_any_overlap(*cleaned):
                side = _direct_outer_side(*cleaned)
                if side < best_side:
                    best, best_side = cleaned, side
        except Exception:
            pass
        if time.monotonic() - started > _DIRECT_TIME_BUDGET_S:
            break
    return best if best is not None else _direct_fallback(hex_num)
