import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

# Enable 64-bit precision for robust geometric calculations
jax.config.update("jax_enable_x64", True)

# --- JAX-JIT COMPILED GEOMETRY KERNELS ---

@jax.jit
def get_unit_hex_verts(center, theta):
    """Calculates vertices of a unit regular hexagon (r=1)."""
    # Vertex angles: 0, 60, 120, 180, 240, 300 degrees
    base_angles = jnp.array([0.0, jnp.pi/3, 2*jnp.pi/3, jnp.pi, 4*jnp.pi/3, 5*jnp.pi/3], dtype=jnp.float64)
    angles = base_angles + theta
    x = jnp.cos(angles)
    y = jnp.sin(angles)
    # Shape: (6, 2)
    return center + jnp.stack([x, y], axis=1)

@jax.jit
def get_container_normals():
    """Returns inward-facing normals for the flat-topped container."""
    # Flat-topped vertices at 0, 60... => Edges at 30, 90...
    # Normal angles: 30, 90, 150, 210, 270, 330
    angles = jnp.array([jnp.pi/6, jnp.pi/2, 5*jnp.pi/6, 7*jnp.pi/6, 3*jnp.pi/2, 11*jnp.pi/6], dtype=jnp.float64)
    nx = jnp.cos(angles)
    ny = jnp.sin(angles)
    return jnp.stack([nx, ny], axis=1)

@jax.jit
def get_hex_normals(theta):
    """Returns normals for a unit hexagon rotated by theta."""
    # Normals are perpendicular to edges. For regular hex, they are at theta + 30, 90...
    angles = jnp.array([jnp.pi/6, jnp.pi/2, 5*jnp.pi/6, 7*jnp.pi/6, 3*jnp.pi/2, 11*jnp.pi/6], dtype=jnp.float64) + theta
    nx = jnp.cos(angles)
    ny = jnp.sin(angles)
    return jnp.stack([nx, ny], axis=1)

@jax.jit
def compute_separation(c1, theta1, c2, theta2):
    """
    Computes the separation distance between two hexagons using SAT.
    Returns positive value if separated, negative if overlapping.
    """
    v1 = get_unit_hex_verts(c1, theta1)
    v2 = get_unit_hex_verts(c2, theta2)
    
    # Candidate Axes: Normals of both hexagons
    n1 = get_hex_normals(theta1)
    n2 = get_hex_normals(theta2)
    axes = jnp.concatenate([n1, n2], axis=0) # (12, 2)
    
    # Project vertices onto axes: (Num_Axes, Num_Verts)
    p1 = jnp.dot(v1, axes.T) 
    p2 = jnp.dot(v2, axes.T)
    
    # Get intervals on each axis
    min1 = jnp.min(p1, axis=0) # (12,)
    max1 = jnp.max(p1, axis=0)
    min2 = jnp.min(p2, axis=0)
    max2 = jnp.max(p2, axis=0)
    
    # Calculate gap on each axis: dist(Interval1, Interval2)
    # Gap > 0 implies separation on that axis.
    # SAT: If Gap > 0 on ANY axis, they are disjoint.
    # We return max(Gap) over all axes.
    gaps = jnp.maximum(min1 - max2, min2 - max1)
    return jnp.max(gaps)

@jax.jit
def containment_violation(c, theta, L):
    """Calculates squared penetration depth outside the container."""
    verts = get_unit_hex_verts(c, theta)
    normals = get_container_normals()
    
    # Distance from center to flat edge of regular hexagon size L:
    # H = L * cos(30) = L * sqrt(3)/2
    H = L * (jnp.sqrt(3.0) / 2.0)
    
    # Project vertices onto container normals
    # Container is intersection of half-planes: x . n <= H
    proj = jnp.dot(verts, normals.T) # (6, 6)
    
    # Violation if proj > H
    diff = proj - H
    return jnp.sum(jax.nn.relu(diff) ** 2)

@jax.jit
def loss_function(params, idx_i, idx_j, penalty_weight, overlap_margin):
    """
    params: Flat array [cx0, cy0, ..., cxN, cyN, theta0, ..., thetaN, L]
    idx_i, idx_j: Indices for pairwise overlap checks.
    """
    # Determine N based on params shape: 3N + 1 = len
    N = (params.shape[0] - 1) // 3
    
    centers = params[:2*N].reshape((N, 2))
    thetas = params[2*N : 3*N]
    L = params[-1]
    
    # 1. Containment Penalty
    cont_v = jax.vmap(containment_violation, in_axes=(0, 0, None))(centers, thetas, L)
    total_cont = jnp.sum(cont_v)
    
    # 2. Overlap Penalty
    # We calculate separation for all pairs
    def pairwise_check(i, j):
        sep = compute_separation(centers[i], thetas[i], centers[j], thetas[j])
        # We enforce separation >= overlap_margin
        # Violation = ReLU(margin - separation)
        return jax.nn.relu(overlap_margin - sep) ** 2
        
    overlaps = jax.vmap(pairwise_check)(idx_i, idx_j)
    total_overlap = jnp.sum(overlaps)
    
    # Total Loss: minimize L subject to constraints
    return L + penalty_weight * (total_cont + total_overlap)

# Compile gradient function
loss_grad = jax.jit(jax.value_and_grad(loss_function))

class Improver:
    def __init__(self, hex_num=11, seed: int = 0):
        self.hex_num = hex_num
        self.seed = seed
        # Precompute pairwise indices for the N hexagons
        self.idx_i, self.idx_j = np.triu_indices(hex_num, k=1)
        # Convert to JAX arrays once
        self.idx_i = jnp.array(self.idx_i)
        self.idx_j = jnp.array(self.idx_j)

    def generate_config(self, seed=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a spiral lattice configuration.
        """
        rng = np.random.RandomState(seed if seed is not None else self.seed)
        
        # Spiral generation
        spacing = 2.05  # Slightly buffered to avoid touching initially
        candidates = []
        ring_limit = 4
        
        # Axial coordinates q, r
        for q in range(-ring_limit, ring_limit + 1):
            for r in range(-ring_limit, ring_limit + 1):
                if -ring_limit <= -q-r <= ring_limit:
                    # Convert to cartesian
                    x = spacing * (q + r/2.0)
                    y = spacing * (np.sqrt(3)/2.0) * r
                    dist_sq = x*x + y*y
                    candidates.append((dist_sq, x, y))
        
        # Sort by distance to center to pack effectively
        candidates.sort(key=lambda k: k[0])
        
        # Select N
        selected = candidates[:self.hex_num]
        centers = np.array([[c[1], c[2]] for c in selected], dtype=np.float64)
        
        # Random angles
        angles = rng.uniform(0, 2*np.pi, size=self.hex_num)
        
        return centers, angles

    def perturb(self, input_config: tuple[np.ndarray, np.ndarray], intensity: float, seed=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies random noise and structural swaps.
        """
        rng = np.random.RandomState(seed if seed is not None else self.seed)
        centers, angles = input_config
        centers = centers.copy()
        angles = angles.copy()
        
        # Gaussian noise
        scale = 0.05 * intensity
        centers += rng.normal(scale=scale, size=centers.shape)
        angles += rng.normal(scale=scale * np.pi, size=angles.shape)
        
        # Structural swaps for higher intensities
        if intensity > 1.0 and self.hex_num > 1:
            n_swaps = int(np.ceil(intensity / 2.0))
            for _ in range(n_swaps):
                i, j = rng.choice(self.hex_num, 2, replace=False)
                centers[i], centers[j] = centers[j], centers[i]
                # Also randomize angles of swapped items
                angles[i] = rng.uniform(0, 2*np.pi)
                angles[j] = rng.uniform(0, 2*np.pi)
                
        return centers, angles

    def improve(self, input_config: tuple[np.ndarray, np.ndarray], seed=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Minimizes enclosing hexagon size L using L-BFGS-B with increasing penalties.
        """
        centers, angles = input_config
        
        # Initial estimate for L
        max_dist = np.max(np.linalg.norm(centers, axis=1))
        initial_L = max(max_dist + 1.5, 3.0)
        
        # Flatten parameters
        x0 = np.concatenate([centers.flatten(), angles, [initial_L]])
        
        # Optimization schedule:
        # Increase weights to transform soft penalties into hard constraints.
        weights = [1.0, 100.0, 10000.0, 1e6, 1e8]
        
        # Safety margin for overlaps (target slightly > 0 separation)
        margin = 2e-3 
        
        current_x = x0.astype(np.float64)
        
        for w in weights:
            # Objective wrapper for Scipy
            def func(x):
                val, grad = loss_grad(x, self.idx_i, self.idx_j, w, margin)
                return float(val), np.array(grad, dtype=np.float64)
            
            # Bounds
            # Centers: loose bounds
            # Angles: unbounded (periodic)
            # L: [1, 50]
            bounds = [(-50, 50)] * (2 * self.hex_num) + \
                     [(None, None)] * self.hex_num + \
                     [(1.0, 50.0)]
            
            res = minimize(
                func,
                current_x,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-7, 'gtol': 1e-7}
            )
            current_x = res.x
            
        # Extract final configuration
        final_centers = current_x[:2*self.hex_num].reshape((self.hex_num, 2))
        final_angles = current_x[2*self.hex_num : 3*self.hex_num]
        final_angles = np.mod(final_angles, 2*np.pi)
        
        return final_centers, final_angles

def entrypoint():
    return Improver