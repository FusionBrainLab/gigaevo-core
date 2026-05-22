"""Per-point block coordinate descent on the sphere — no surrogate, no JAX.

This is a structurally non-gradient method. We sweep through the n points in
a random order; for each point i, we freeze the other n-1 points and update
x_i alone to reduce its row-max inner product. The update is a few Riemannian
gradient steps on the 1-point subproblem (LSE over a single row of G).

Per-point work is O(d) for the gradient plus O(n) for the row inner
products. A full sweep is O(n²d). At n=600, d=11 a sweep is ~7e6 flops —
faster than one global LSE gradient evaluation.

improve: ~20 random-order sweeps × 6 RGD steps per point at low alpha (200),
then ~10 sweeps at high alpha (2000).
perturb: pick K worst-row-max points; replace each with a fresh random
direction tangent to a random unit vector (large reset). K and step size
scale with intensity.
generate_config: Gaussian on the sphere.
"""

from __future__ import annotations

import numpy as np


def _renormalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


def _single_point_step(
    x_i: np.ndarray,
    others_dots: np.ndarray,
    others: np.ndarray,
    alpha: float,
    step: float,
) -> tuple[np.ndarray, float]:
    """One Riemannian gradient step on the 1-point LSE subproblem.

    others_dots[j] = <x_j, x_i> for j ≠ i (precomputed, length n-1).
    others[j]      = x_j  (matrix of shape (n-1, d)).

    Returns the new x_i and the new local max <x_j, x_i> (for diagnostics).
    """
    m = float(others_dots.max())
    e = np.exp(alpha * (others_dots - m))
    z = float(e.sum())
    if z < 1e-300:
        return x_i, m
    w = e / z  # softmax weights, sum to 1
    # Euclidean grad of (1/alpha) log Σ exp(alpha <x_j, x_i>) w.r.t. x_i
    g_E = (w[:, None] * others).sum(axis=0)
    # Riemannian tangent projection
    tan = g_E - float(g_E @ x_i) * x_i
    tan_norm = float(np.linalg.norm(tan))
    if tan_norm < 1e-14:
        return x_i, m
    direction = tan / tan_norm
    angle = step * tan_norm
    new_xi = np.cos(angle) * x_i - np.sin(angle) * direction
    return new_xi, m


class Improver:
    def __init__(self, n: int, d: int, seed: int = 0):
        self.n = int(n)
        self.d = int(d)
        self.seed = int(seed)

    def generate_config(self, seed=None) -> np.ndarray:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        X = rng.standard_normal((self.n, self.d)).astype(np.float64)
        return _renormalize(X)

    def improve(self, points: np.ndarray, seed=None) -> np.ndarray:
        X = _renormalize(np.asarray(points, dtype=np.float64).copy())
        rng = np.random.default_rng(seed if seed is not None else self.seed)

        def global_mu(M):
            G = M @ M.T
            np.fill_diagonal(G, -np.inf)
            return float(G.max())

        # Two-phase α schedule. Steps are intentionally small — single-row
        # LSE descent is NOT a descent direction on global μ, so we wrap each
        # sweep in a sweep-level Armijo accept/reject on the global maximum.
        # If a sweep increases μ, revert and halve the step.
        phases = [
            {"alpha": 200.0, "max_sweeps": 25, "inner": 4, "step": 0.08},
            {"alpha": 2000.0, "max_sweeps": 15, "inner": 3, "step": 0.03},
        ]

        for phase in phases:
            alpha = phase["alpha"]
            step = phase["step"]
            inner = phase["inner"]
            stalled = 0
            for _ in range(phase["max_sweeps"]):
                mu_before = global_mu(X)
                X_trial = X.copy()
                order = rng.permutation(self.n)
                idx_all = np.arange(self.n)
                for i in order:
                    x_i = X_trial[i]
                    dots = X_trial @ x_i
                    dots[i] = -np.inf
                    keep = idx_all != i
                    others = X_trial[keep]
                    others_dots = dots[keep]
                    for _ in range(inner):
                        x_i, _m = _single_point_step(
                            x_i, others_dots, others, alpha, step
                        )
                        others_dots = others @ x_i
                    X_trial[i] = x_i
                mu_after = global_mu(X_trial)
                if mu_after < mu_before - 1e-9:
                    X = X_trial
                    stalled = 0
                else:
                    step *= 0.5
                    stalled += 1
                    if stalled >= 3 or step < 1e-4:
                        break
        return _renormalize(X)

    def perturb(self, points: np.ndarray, intensity: float, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        X = _renormalize(np.asarray(points, dtype=np.float64).copy())
        intensity = float(np.clip(intensity, 0.0, 1.0))

        # K worst points get a hard reset to a random direction; remaining ones
        # are untouched. K scales with intensity (1 → ~12% of n).
        K = max(1, int(np.ceil(self.n * 0.12 * intensity)))
        K = min(K, self.n - 1)
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        worst = np.argpartition(-G.max(axis=1), K - 1)[:K]

        # Replace each worst point with a fresh uniform-sphere sample, mixed
        # with the original by an angle proportional to intensity. At
        # intensity=1 we fully reset; at low intensity we slightly perturb.
        fresh = rng.standard_normal((K, self.d)).astype(np.float64)
        fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
        mix_angle = intensity * (np.pi / 2.0)
        cos_a, sin_a = float(np.cos(mix_angle)), float(np.sin(mix_angle))
        # Ensure fresh is tangent-ish: subtract the X-component to slerp.
        # Then renormalize the per-row interpolation.
        X[worst] = cos_a * X[worst] + sin_a * fresh
        X[worst] = _renormalize(X[worst])

        return _renormalize(X)


def entrypoint():
    return Improver
