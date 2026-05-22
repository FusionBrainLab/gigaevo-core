"""Riemannian gradient descent on the oblique manifold (S^(d-1))^n.

The configuration X ∈ R^(n×d) lives on the product of n unit spheres. The
objective is the LogSumExp surrogate of max_{i<j} <x_i, x_j>, but every step
respects the manifold geometry: the gradient is projected onto the tangent
bundle and the update is an exact exponential-map retraction. No Euclidean
over-parameterization, no division by ambient norm, no radial gradient
collapse.

improve: 5-stop alpha anneal (60 → 3000), 80 RGD steps per stop with
backtracking Armijo line search.
perturb: identify the K most-active points (highest row-max of Gram), push
each along its local tangent steepest-ascent direction by an angle scaled
by `intensity`; everything else stays put.
generate_config: Gaussian on the sphere, deterministic from `seed`.
"""

from __future__ import annotations

import numpy as np


def _lse_value_grad(X: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    """Pair-once-counted LSE of max_{i<j} <x_i, x_j> with stable max-shift.

    Returns (value, Euclidean gradient w.r.t. X). The gradient W @ X is built
    from the symmetric softmax weight matrix W (with diagonal masked out).
    """
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -np.inf)
    m = float(G.max())
    G += 0.0  # no-op; keep dtype
    E = np.exp(alpha * (G - m))
    np.fill_diagonal(E, 0.0)
    Z_full = float(E.sum())
    if Z_full < 1e-300:
        return m, np.zeros_like(X)
    # f = m + (1/alpha) log(Z_full / 2)
    value = m + np.log(0.5 * Z_full) / alpha
    W = (2.0 / Z_full) * E
    grad = W @ X
    return value, grad


def _project_tangent(g_E: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-point tangent projection on the oblique manifold."""
    radial = np.sum(g_E * X, axis=1, keepdims=True)
    return g_E - radial * X


def _retract(X: np.ndarray, V: np.ndarray, step: float) -> np.ndarray:
    """Exponential-map retraction along tangent V (per row geodesic).

    Each row: x_k(t) = cos(t‖v_k‖) x_k + sin(t‖v_k‖) v_k / ‖v_k‖.
    """
    V_norms = np.linalg.norm(V, axis=1, keepdims=True)
    safe = np.where(V_norms > 1e-30, V_norms, 1.0)
    angles = step * V_norms
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    direction = V / safe
    return cos_a * X + sin_a * direction


def _renormalize(X: np.ndarray) -> np.ndarray:
    """Strict float64 row-normalization to satisfy the 1e-12 tolerance."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


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
        X = np.asarray(points, dtype=np.float64).copy()
        X = _renormalize(X)

        alphas = (60.0, 200.0, 600.0, 1500.0, 3000.0)
        steps_per_alpha = 80

        for alpha in alphas:
            step = 0.5  # initial Riemannian step (radians)
            f_prev, grad_E = _lse_value_grad(X, alpha)
            tan = _project_tangent(grad_E, X)
            for _ in range(steps_per_alpha):
                tan_norm = float(np.linalg.norm(tan))
                if tan_norm < 1e-14:
                    break
                # Armijo backtracking: shrink step until f decreases.
                for _bt in range(20):
                    X_try = _retract(X, -tan, step)
                    f_try, grad_try_E = _lse_value_grad(X_try, alpha)
                    if f_try < f_prev - 1e-12:
                        X = X_try
                        f_prev = f_try
                        tan = _project_tangent(grad_try_E, X)
                        step = min(step * 1.5, 1.0)
                        break
                    step *= 0.5
                else:
                    break  # backtracking failed → exit this alpha
        return _renormalize(X)

    def perturb(self, points: np.ndarray, intensity: float, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        X = np.asarray(points, dtype=np.float64).copy()
        X = _renormalize(X)

        intensity = float(np.clip(intensity, 0.0, 1.0))
        K = max(2, int(2 + 60 * intensity))
        K = min(K, self.n)

        # Locate the K points participating in the largest pairwise inner products.
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        row_max = G.max(axis=1)
        active = np.argpartition(-row_max, K - 1)[:K]

        # Push each active point along its own tangent steepest-ascent direction
        # (i.e., away from its current worst neighbour) by a random angle scaled
        # by intensity. Untouched points stay exactly put.
        for idx in active:
            j = int(np.argmax(G[idx]))
            push = X[j] - float(X[j] @ X[idx]) * X[idx]  # tangent of x_j at x_i
            push_norm = float(np.linalg.norm(push))
            if push_norm < 1e-12:
                # Degenerate (points coincident); pick a random tangent direction.
                v = rng.standard_normal(self.d).astype(np.float64)
                v -= float(v @ X[idx]) * X[idx]
                push_norm = float(np.linalg.norm(v))
                if push_norm < 1e-12:
                    continue
                push = v
            push /= push_norm
            angle = intensity * float(rng.uniform(0.05, 0.5))
            X[idx] = np.cos(angle) * X[idx] - np.sin(angle) * push

        return _renormalize(X)


def entrypoint():
    return Improver
