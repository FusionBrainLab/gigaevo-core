import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

class Improver:
    """
    Optimizes the packing of N unit regular hexagons into the smallest possible
    flat-topped regular enclosing hexagon.
    """

    def __init__(self, hex_num=11, seed: int = 0):
        self.hex_num = hex_num
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # --- JAX Compilation ---
        self._setup_geometric_constants()
        self._compile_jax_functions()

    def _setup_geometric_constants(self):
        """Pre-calculate constant geometry for unit and container hexagons."""
        # Unit Hexagon Vertices (radius 1)
        # Angles: 0, 60, 120...
        angles_k = jnp.linspace(0, 2*jnp.pi, 7)[:-1]
        self.unit_verts = jnp.stack([jnp.cos(angles_k), jnp.sin(angles_k)], axis=1)
        
        # Container Normals (Flat-topped)
        # Vertices at 0, 60... => Faces/Normals at 30, 90, 150...
        # Constraint: v . n <= L * sqrt(3)/2
        cont_angles = angles_k + jnp.pi/6
        self.cont_normals = jnp.stack([jnp.cos(cont_angles), jnp.sin(cont_angles)], axis=1)
        
        # SAT Axes Base (3 unique axes for a regular hexagon)
        # Normals at 30, 90, 150 relative to rotation 0
        self.sat_base_axes = jnp.array([
            [jnp.cos(jnp.pi/6), jnp.sin(jnp.pi/6)],
            [jnp.cos(jnp.pi/2), jnp.sin(jnp.pi/2)],
            [jnp.cos(5*jnp.pi/6), jnp.sin(5*jnp.pi/6)]
        ])

    def _compile_jax_functions(self):
        """Defines and compiles JAX functions for the optimization loop."""
        
        # --- Helpers ---
        def get_hex_verts(center, theta):
            c, s = jnp.cos(theta), jnp.sin(theta)
            R = jnp.array([[c, -s], [s, c]])
            return (self.unit_verts @ R.T) + center

        def get_all_verts(centers, angles):
            return jax.vmap(get_hex_verts)(centers, angles)

        def get_container_L(all_verts):
            # Project all vertices onto container normals
            # pts: (N*6, 2)
            pts = all_verts.reshape(-1, 2)
            projections = pts @ self.cont_normals.T
            max_proj = jnp.max(projections)
            # L = max_proj / (sqrt(3)/2)
            return max_proj * (2.0 / jnp.sqrt(3.0))

        # --- SAT Overlap ---
        def pairwise_overlap(c1, a1, c2, a2, margin):
            # Generate 6 axes (3 from each hex)
            def get_axes(theta):
                c, s = jnp.cos(theta), jnp.sin(theta)
                R = jnp.array([[c, -s], [s, c]])
                return self.sat_base_axes @ R.T
            
            axes = jnp.concatenate([get_axes(a1), get_axes(a2)], axis=0)
            
            v1 = get_hex_verts(c1, a1)
            v2 = get_hex_verts(c2, a2)
            
            # Project onto axes to get intervals
            def project(verts, axis):
                p = verts @ axis
                return jnp.min(p), jnp.max(p)
            
            # Vectorized projection
            min1, max1 = jax.vmap(partial(project, v1))(axes)
            min2, max2 = jax.vmap(partial(project, v2))(axes)
            
            # Overlap on each axis
            # Add margin: we want effective overlap to be (overlap + margin)
            # If (max1 - min2) > -margin, etc.
            # Real overlap depth:
            ov = jnp.minimum(max1, max2) - jnp.maximum(min1, min2)
            
            # If any axis has separation (ov < -margin), strictly no collision
            # We use soft logic for gradients:
            # We want to penalize if ALL axes have ov > -margin
            
            # Smooth approximation for min: SoftMin or just Min
            # Strict SAT: penetration = min(ov). If penetration < 0, separated.
            # We want to penalize positive penetration.
            
            penetration = jnp.min(ov)
            
            # Loss = ReLU(penetration + margin)
            return jnp.maximum(0.0, penetration + margin)

        def total_overlap_loss(centers, angles, margin):
            N = centers.shape[0]
            idx_i, idx_j = jnp.triu_indices(N, 1)
            
            def body(i, j):
                return pairwise_overlap(centers[i], angles[i], centers[j], angles[j], margin)
            
            overlaps = jax.vmap(body)(idx_i, idx_j)
            return jnp.sum(overlaps)

        # --- Main Loss ---
        def loss_fn(params, w_overlap, w_container, margin):
            centers, angles = params
            all_verts = get_all_verts(centers, angles)
            
            L = get_container_L(all_verts)
            ov_loss = total_overlap_loss(centers, angles, margin)
            
            return w_container * L + w_overlap * ov_loss, (L, ov_loss)

        self.loss_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        self.check_overlap = jax.jit(total_overlap_loss)

    def generate_config(self, seed=None) -> tuple[np.ndarray, np.ndarray]:
        """Generates a valid honeycomb grid configuration."""
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        
        # Grid parameters
        dx = np.sqrt(3)
        dy = 1.5
        
        points = []
        grid_r = int(np.sqrt(self.hex_num)) + 2
        for r in range(-grid_r, grid_r + 1):
            for q in range(-grid_r, grid_r + 1):
                # Offset coordinates
                x = q * dx + (r % 2) * (dx / 2)
                y = r * dy
                points.append([x, y])
        
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        points = points[np.argsort(dists)]
        
        centers = points[:self.hex_num]
        
        # Add tiny noise to avoid singular gradients
        centers += rng.normal(0, 0.001, size=centers.shape)
        # Random rotations
        angles = rng.uniform(0, 2*np.pi, size=self.hex_num)
        
        return centers, angles

    def perturb(self, input_config, intensity, seed=None):
        rng = np.random.default_rng(seed)
        centers, angles = input_config
        centers = centers.copy()
        angles = angles.copy()
        
        if intensity > 10.0:
            # Re-shuffle
            idx = rng.permutation(self.hex_num)
            centers = centers[idx]
            angles = rng.uniform(0, 2*np.pi, size=self.hex_num)
            centers *= 1.1 # Expand
        elif intensity > 1.0:
            centers += rng.normal(0, 0.2, size=centers.shape)
            angles += rng.normal(0, 0.5, size=angles.shape)
        else:
            centers += rng.normal(0, 0.05, size=centers.shape)
            angles += rng.normal(0, 0.1, size=angles.shape)
            
        return centers, angles

    def improve(self, input_config, seed=None):
        centers_np, angles_np = input_config
        centers = jnp.array(centers_np)
        angles = jnp.array(angles_np)
        
        # --- Phase 1: Gradient Descent ---
        
        params = (centers, angles)
        
        # Hyperparams
        steps = 400
        lr = 0.05
        # Adam state
        m = (jnp.zeros_like(centers), jnp.zeros_like(angles))
        v = (jnp.zeros_like(centers), jnp.zeros_like(angles))
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        # We use a margin > 0 so the optimizer creates a gap.
        # This prevents "touching" from becoming "overlapping" due to float errors.
        margin = 0.01 
        w_overlap = 500.0
        w_container = 1.0
        
        for i in range(steps):
            (loss, (L, ov)), grads = self.loss_and_grad(params, w_overlap, w_container, margin)
            
            # Simple Adam update
            c_g, a_g = grads
            c_p, a_p = params
            m_c, m_a = m
            v_c, v_a = v
            
            # Centers
            m_c = beta1 * m_c + (1-beta1)*c_g
            v_c = beta2 * v_c + (1-beta2)*(c_g**2)
            m_c_hat = m_c / (1 - beta1**(i+1))
            v_c_hat = v_c / (1 - beta2**(i+1))
            c_p = c_p - lr * m_c_hat / (jnp.sqrt(v_c_hat) + eps)
            
            # Angles
            m_a = beta1 * m_a + (1-beta1)*a_g
            v_a = beta2 * v_a + (1-beta2)*(a_g**2)
            m_a_hat = m_a / (1 - beta1**(i+1))
            v_a_hat = v_a / (1 - beta2**(i+1))
            a_p = a_p - lr * m_a_hat / (jnp.sqrt(v_a_hat) + eps)
            
            params = (c_p, a_p)
            m = (m_c, m_a)
            v = (v_c, v_a)
            
            if i % 100 == 0:
                lr *= 0.8 # Decay
                
        # --- Phase 2: Post-Correction (NumPy) ---
        # Strictly enforce validity using CPU geometric checks
        
        final_c = np.array(params[0])
        final_a = np.array(params[1])
        
        final_c, final_a = self._resolve_overlaps(final_c, final_a)
        
        return final_c, final_a

    def _resolve_overlaps(self, centers, angles):
        """
        Iterative solver to mechanically push overlapping hexagons apart.
        Ensures strict validity at the cost of potential minor expansion.
        """
        
        # Local helper for strict SAT (no JAX overhead)
        def get_verts(c, t):
            # Unit hex verts
            thetas = np.linspace(0, 2*np.pi, 7)[:-1]
            v = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
            # Rotate
            cos_t, sin_t = np.cos(t), np.sin(t)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            return (v @ R.T) + c

        def check_pair(c1, t1, c2, t2):
            # Axes: 30, 90, 150 relative to t1 and t2
            base = np.array([np.pi/6, np.pi/2, 5*np.pi/6])
            
            axes1 = np.stack([np.cos(base+t1), np.sin(base+t1)], axis=1)
            axes2 = np.stack([np.cos(base+t2), np.sin(base+t2)], axis=1)
            axes = np.concatenate([axes1, axes2])
            
            v1 = get_verts(c1, t1)
            v2 = get_verts(c2, t2)
            
            min_overlap = float('inf')
            sep_axis = None
            
            for ax in axes:
                p1 = v1 @ ax
                p2 = v2 @ ax
                
                # Check separation
                if np.max(p1) < np.min(p2) or np.max(p2) < np.min(p1):
                    return False, 0.0, None # Separated
                
                # Overlap depth
                o = min(np.max(p1), np.max(p2)) - max(np.min(p1), np.min(p2))
                if o < min_overlap:
                    min_overlap = o
                    # Direction: verify pushing c1 away from c2
                    if np.dot(c1 - c2, ax) > 0:
                        sep_axis = ax
                    else:
                        sep_axis = -ax
            
            return True, min_overlap, sep_axis

        # Iterative solver
        for _ in range(50): # Max iterations
            overlap_found = False
            for i in range(self.hex_num):
                for j in range(i+1, self.hex_num):
                    # Fast dist check
                    if np.linalg.norm(centers[i] - centers[j]) > 2.05:
                        continue
                        
                    is_ov, depth, axis = check_pair(centers[i], angles[i], centers[j], angles[j])
                    if is_ov and depth > 1e-7:
                        overlap_found = True
                        # Push apart
                        # Move each by half depth + epsilon
                        push = axis * (depth * 0.51) 
                        centers[i] += push
                        centers[j] -= push
            
            if not overlap_found:
                break
                
        return centers, angles

def entrypoint():
    return Improver