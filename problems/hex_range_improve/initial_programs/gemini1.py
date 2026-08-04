import jax
import jax.numpy as jnp
import numpy as np
import optax

# ==========================================
# GEOMETRY CONSTANTS & CONFIG
# ==========================================

# Enable 64-bit precision for strict geometric adherence
jax.config.update("jax_enable_x64", True)

SQRT3 = np.sqrt(3.0)

# Unit Hexagon: Radius 1. Vertices at 0, 60, ...
_THETA_UNIT = jnp.linspace(0, 2 * jnp.pi, 7)[:-1]
_UNIT_VERTS_BASE = jnp.stack([jnp.cos(_THETA_UNIT), jnp.sin(_THETA_UNIT)], axis=1)

# Container (Flat-topped): Normals at 30, 90...
_THETA_NORM = jnp.linspace(jnp.pi / 6, 2 * jnp.pi + jnp.pi / 6, 7)[:-1]
_CONT_NORMALS = jnp.stack([jnp.cos(_THETA_NORM), jnp.sin(_THETA_NORM)], axis=1)

# SAT Axes Base: 30, 90, 150 (Unique normals relative to rotation)
_SAT_AXES_BASE = jnp.array([jnp.pi / 6, jnp.pi / 2, 5 * jnp.pi / 6])


@jax.jit
def get_verts(c, t):
    """Computes vertices for a unit hexagon at center c with rotation t."""
    cos_t, sin_t = jnp.cos(t), jnp.sin(t)
    # Rotate base vertices and translate
    vx = _UNIT_VERTS_BASE[:, 0] * cos_t - _UNIT_VERTS_BASE[:, 1] * sin_t + c[0]
    vy = _UNIT_VERTS_BASE[:, 0] * sin_t + _UNIT_VERTS_BASE[:, 1] * cos_t + c[1]
    return jnp.stack([vx, vy], axis=1)


@jax.jit
def pairwise_overlap_penalty(c1, t1, c2, t2, margin):
    """
    Computes SAT overlap penalty with a safety margin.
    Penalty > 0 if distance between hexes < touching_distance + margin.
    """
    # 1. Candidate Axes (3 from hex1, 3 from hex2)
    a1 = t1 + _SAT_AXES_BASE
    a2 = t2 + _SAT_AXES_BASE

    ax1 = jnp.stack([jnp.cos(a1), jnp.sin(a1)], axis=1)
    ax2 = jnp.stack([jnp.cos(a2), jnp.sin(a2)], axis=1)
    axes = jnp.concatenate([ax1, ax2])  # Shape (6, 2)

    # 2. Project Centers
    p1 = jnp.dot(axes, c1)
    p2 = jnp.dot(axes, c2)
    dist_centers = jnp.abs(p1 - p2)

    # 3. Project Hexagon "Radii"
    # Create local vertices for projection
    v1_x = _UNIT_VERTS_BASE[:, 0] * jnp.cos(t1) - _UNIT_VERTS_BASE[:, 1] * jnp.sin(t1)
    v1_y = _UNIT_VERTS_BASE[:, 0] * jnp.sin(t1) + _UNIT_VERTS_BASE[:, 1] * jnp.cos(t1)
    v1_loc = jnp.stack([v1_x, v1_y], axis=1)

    v2_x = _UNIT_VERTS_BASE[:, 0] * jnp.cos(t2) - _UNIT_VERTS_BASE[:, 1] * jnp.sin(t2)
    v2_y = _UNIT_VERTS_BASE[:, 0] * jnp.sin(t2) + _UNIT_VERTS_BASE[:, 1] * jnp.cos(t2)
    v2_loc = jnp.stack([v2_x, v2_y], axis=1)

    # Radius on axis is max absolute projection of local vertices
    r1 = jnp.max(jnp.abs(jnp.dot(v1_loc, axes.T)), axis=0)
    r2 = jnp.max(jnp.abs(jnp.dot(v2_loc, axes.T)), axis=0)

    # 4. Check Separation
    # Gap = CenterDist - (r1 + r2)
    # Separation is the BEST gap found (max over axes)
    # If Separation > margin, we are safe.
    gaps = dist_centers - (r1 + r2)
    separation = jnp.max(gaps)

    # Penalty if separation < margin
    return jax.nn.relu(margin - separation) ** 2


@jax.jit
def loss_fn(params, weights, indices):
    centers, angles, l_log = params
    L = jnp.exp(l_log)
    w_contain, w_overlap, w_repel, margin = weights
    idx_i, idx_j = indices

    # --- Containment ---
    # Container apothem = L * sqrt(3)/2
    c_apothem = L * (SQRT3 / 2.0)

    # Vertices of all hexes
    all_verts = jax.vmap(get_verts)(centers, angles)
    all_verts_flat = all_verts.reshape(-1, 2)

    # Project onto container normals
    dists = jnp.dot(all_verts_flat, _CONT_NORMALS.T)

    # Violation if dist > apothem
    violations = jax.nn.relu(dists - c_apothem)
    loss_c = jnp.sum(violations**2)

    # --- Overlap ---
    # Vectorized pairwise checks
    c1, t1 = centers[idx_i], angles[idx_i]
    c2, t2 = centers[idx_j], angles[idx_j]

    overlap_penalties = jax.vmap(pairwise_overlap_penalty, in_axes=(0, 0, 0, 0, None))(
        c1, t1, c2, t2, margin
    )
    loss_o = jnp.sum(overlap_penalties)

    # --- Repulsion ---
    # Short-range soft repulsion to prevent singularity at dist=0
    d2 = jnp.sum((c1 - c2) ** 2, axis=1)
    loss_r = jnp.sum(jax.nn.relu(0.01 - d2))

    # Total Loss
    return L + w_contain * loss_c + w_overlap * loss_o + w_repel * loss_r


# Gradient Function
loss_grad_fn = jax.value_and_grad(loss_fn)


class Improver:
    def __init__(self, hex_num=11, seed: int = 0):
        self.hex_num = hex_num
        self.seed = seed

        # Precompute indices for pairwise interactions to keep JIT happy
        self.idx_i, self.idx_j = jnp.triu_indices(self.hex_num, 1)

    def generate_config(self, seed=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a valid initial configuration using a spiral lattice packing.
        """
        s = seed if seed is not None else self.seed
        rng = np.random.default_rng(s)

        # Generate hex grid points
        grid = []
        dim = int(np.sqrt(self.hex_num)) + 3
        for q in range(-dim, dim + 1):
            for r in range(-dim, dim + 1):
                # Cartesian conversion
                x = SQRT3 * (q + r / 2.0)
                y = 1.5 * r
                dist_sq = x * x + y * y
                grid.append(((x, y), dist_sq))

        # Sort by distance from center
        grid.sort(key=lambda item: item[1])

        centers = []
        # Use spacing > 1.732 to strictly avoid initial overlap
        spacing = 1.75
        for i in range(self.hex_num):
            pt = grid[i][0]
            centers.append([pt[0] * (spacing / SQRT3), pt[1] * (spacing / 1.5)])

        centers = np.array(centers)

        # Random angles
        angles = rng.uniform(0, 2 * np.pi, self.hex_num)

        return centers, angles

    def improve(
        self, input_config: tuple[np.ndarray, np.ndarray], seed=None
    ) -> tuple[np.ndarray, np.ndarray]:
        c_in, a_in = input_config

        # Validation
        if c_in.shape != (self.hex_num, 2) or not np.all(np.isfinite(c_in)):
            c_in, a_in = self.generate_config(seed)

        # JAX conversion
        # Initialize L (log space)
        # Start with a safe L
        current_r = np.max(np.linalg.norm(c_in, axis=1)) + 2.0
        params = (jnp.array(c_in), jnp.array(a_in), jnp.array(np.log(current_r)))

        # Optimization Setup
        # Schedule: 1500 steps.
        # Phase 1 (0-500): Soft weights, high learning rate. Untangling.
        # Phase 2 (500-1200): Ramping weights.
        # Phase 3 (1200-1500): Hard weights, fine tuning.

        iterations = 1500
        schedule = optax.piecewise_constant_schedule(
            init_value=1e-2, boundaries_and_scales={500: 0.5, 1000: 0.1, 1300: 0.1}
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate=schedule)
        )
        opt_state = optimizer.init(params)

        # Capture indices for JIT
        ii, jj = self.idx_i, self.idx_j

        @jax.jit
        def step(p, st, w):
            # w = [w_contain, w_overlap, w_repel, margin]
            val, grads = loss_grad_fn(p, w, (ii, jj))
            updates, new_st = optimizer.update(grads, st)
            new_p = optax.apply_updates(p, updates)
            return new_p, new_st, val

        curr_params = params
        curr_state = opt_state

        for i in range(iterations):
            # Dynamic Weights
            if i < 500:
                # Phase 1: Relaxed
                w_contain = 10.0
                w_overlap = 10.0
                margin = 0.0  # No margin yet
            else:
                # Phase 2/3: Tightening
                progress = (i - 500) / 1000.0
                # Exponential ramp to 1e6
                scale = 10.0 ** (6.0 * progress)
                w_contain = 10.0 * scale
                w_overlap = 10.0 * scale
                margin = 1e-3  # Enforce safety margin

            w_repel = 100.0 if i < 200 else 0.0

            weights = jnp.array([w_contain, w_overlap, w_repel, margin])

            curr_params, curr_state, loss = step(curr_params, curr_state, weights)

            if jnp.isnan(loss):
                # Reset if divergence
                return self.perturb(input_config, 50.0, seed)

        final_c, final_a, final_l_log = curr_params
        return np.array(final_c), np.array(final_a)

    def perturb(
        self, input_config: tuple[np.ndarray, np.ndarray], intensity: float, seed=None
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        c, a = input_config
        c = c.copy()
        a = a.copy()

        # High intensity reset
        if intensity > 40.0:
            return self.generate_config(seed)

        # 1. Jitter
        c += rng.normal(0, 0.05 * intensity, c.shape)
        a += rng.normal(0, 0.2 * intensity, a.shape)

        # 2. Swap
        if intensity > 2.0:
            idx = rng.permutation(self.hex_num)
            i1, i2 = idx[0], idx[1]
            c[i1], c[i2] = c[i2], c[i1].copy()
            a[i1], a[i2] = a[i2], a[i1].copy()

        # 3. Expansion
        if intensity > 10.0:
            c *= 1.1

        return c, a


def entrypoint():
    return Improver
