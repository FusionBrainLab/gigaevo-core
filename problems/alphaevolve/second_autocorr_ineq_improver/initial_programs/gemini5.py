import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve
from scipy.interpolate import interp1d

class Improver:
    def __init__(self, seed: int = 0):
        """
        Initialize with reproducible random state.
        """
        self.seed = seed
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

    def _objective_and_grad(self, f):
        """
        Computes -C(f) and its gradient with respect to f.
        Minimizing this maximizes C(f).
        """
        # Ensure f is float64 for precision
        f = f.astype(np.float64)
        N = len(f)
        
        # 1. Forward pass: Compute g = f * f
        # Use valid/full modes carefully. 
        # f has shape (N,), g has shape (2N-1,)
        g = convolve(f, f, mode='full')
        
        # Compute norms
        g_sq = g * g
        A = np.sum(g_sq)              # ||g||_2^2
        sum_f = np.sum(f)
        B = sum_f * sum_f             # ||g||_1 = ||f||_1^2
        
        k_max = np.argmax(g)
        M = g[k_max]                  # ||g||_inf
        
        # Avoid division by zero
        if B == 0 or M == 0:
            return 0.0, np.zeros_like(f)
        
        denom = B * M
        C = A / denom
        
        # 2. Backward pass: Compute gradients
        # We want grad of (-C) = - grad(C)
        # grad(C) = (1/BM) * grad(A) - (A/B^2 M) * grad(B) - (A/BM^2) * grad(M)
        
        # factor1 corresponds to term involving grad A
        factor_A = 1.0 / denom
        
        # factor2 corresponds to term involving grad B
        factor_B = -A / (B * denom)
        
        # factor3 corresponds to term involving grad M
        factor_M = -A / (M * denom)
        
        # -- Gradient of A --
        # dA/df_i = 4 * sum_k g[k] * f[k-i]
        # This is 4 * cross_correlation(g, f)
        # computed as convolve(g, f[::-1], mode='valid')
        grad_A = 4.0 * convolve(g, f[::-1], mode='valid')
        
        # -- Gradient of B --
        # dB/df_i = 2 * sum(f)
        grad_B = np.full(N, 2.0 * sum_f)
        
        # -- Gradient of M --
        # M = g[k_max] = sum_j f[j] * f[k_max - j]
        # dM/df_i = 2 * f[k_max - i]
        # We need to extract the vector corresponding to shifted f
        # Indices of f that contribute are j such that k_max - j is a valid index for f
        # i.e., 0 <= k_max - i < N  =>  k_max - N < i <= k_max
        # We can construct this vector by slicing
        grad_M = np.zeros(N)
        
        # range of i where f[k_max - i] is valid
        # Let j = k_max - i. We need 0 <= j < N.
        # => 0 <= k_max - i < N
        # => i <= k_max  AND  i > k_max - N
        start_i = max(0, k_max - N + 1)
        end_i = min(N, k_max + 1)
        
        # The values are f[k_max - i]
        # Slice f from (k_max - end_i + 1) to (k_max - start_i + 1) reversed?
        # Easier to iterate or use smart slicing. 
        # k_max - i goes from (k_max - start_i) down to (k_max - (end_i - 1))
        if start_i < end_i:
            indices = k_max - np.arange(start_i, end_i)
            grad_M[start_i:end_i] = 2.0 * f[indices]
            
        # Combine gradients
        grad = factor_A * grad_A + factor_B * grad_B + factor_M * grad_M
        
        return -C, -grad

    def improve(self, input_f: np.ndarray) -> np.ndarray:
        """
        Refines an existing function f using rigorous local optimization.
        Uses L-BFGS-B with analytical gradients.
        Adopts a multigrid-like approach: if N is small, upsample first.
        """
        N = len(input_f)
        f = input_f.copy()
        
        # Target resolutions for multigrid steps
        # If N is very large, just optimize. If small, upscale.
        target_N = max(N, 1024)
        
        # If input is significantly smaller than target, upsample in steps
        while len(f) <= target_N / 2:
            # Optimize at current resolution
            res = minimize(
                self._objective_and_grad, 
                f, 
                method='L-BFGS-B', 
                jac=True, 
                bounds=[(0, None)] * len(f),
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            f = res.x
            
            # Upsample
            new_len = len(f) * 2
            x_old = np.linspace(0, 1, len(f))
            x_new = np.linspace(0, 1, new_len)
            f = interp1d(x_old, f, kind='linear', fill_value="extrapolate")(x_new)
            # Add slight perturbation to break symmetries
            f += self.rng.normal(0, 1e-4 * np.max(f), size=new_len)
            f = np.abs(f)
            
        # Final optimization at target resolution
        # For very large N (>10000), we might limit iterations to fit time
        max_iter = 2000 if len(f) < 5000 else 100
        
        res = minimize(
            self._objective_and_grad, 
            f, 
            method='L-BFGS-B', 
            jac=True, 
            bounds=[(0, None)] * len(f),
            options={'maxiter': max_iter, 'ftol': 1e-9}
        )
        
        return res.x

    def perturb(self, input_f: np.ndarray, intensity: float) -> np.ndarray:
        """
        Applies DISCRETE, structural, or RESOLUTION modifications.
        intensity: controls noise magnitude.
        """
        f = input_f.copy()
        N = len(f)
        
        # 1. Additive Noise (High frequency)
        noise = self.rng.normal(0, intensity * np.max(f) * 0.1, size=N)
        f = f + noise
        
        # 2. Structural perturbation (Low frequency modulation)
        # Multiply by a slow sine wave
        x = np.linspace(0, 1, N)
        freq = self.rng.integers(1, 5)
        phase = self.rng.uniform(0, 2*np.pi)
        mod = 1.0 + intensity * 0.2 * np.sin(2 * np.pi * freq * x + phase)
        f = f * mod
        
        return np.abs(f)

    def generate_config(self, initial_resolution: int = 1000) -> np.ndarray:
        """
        Generates a valid starting function f with a maximum possible fitness.
        Known good shape: f(x) ~ x^(-0.5) near 0.
        We initialize with a tuned singular function profile.
        """
        N = initial_resolution
        
        # Construct a candidate based on f(x) = x^(-0.52) 
        # This profile is known to yield high C values (~0.8-0.9).
        # We shift x slightly to avoid division by zero and tune the exponent.
        x = np.linspace(0, 1, N)
        # Using a small offset for the singularity
        offset = 0.5 / N
        exponent = 0.5  # Starting close to 1/sqrt(x)
        
        # We create a mix of the singular function and a small constant background
        f = 1.0 / np.power(x + offset, exponent)
        
        # Normalize roughly (optional, but good for optimizer scale)
        f = f / np.max(f)
        
        # Apply a quick initial optimization to settle the shape
        # This ensures the returned config is already high quality
        # We run a short improvement step
        f = self.improve(f)
        
        return f

def entrypoint():
    return Improver