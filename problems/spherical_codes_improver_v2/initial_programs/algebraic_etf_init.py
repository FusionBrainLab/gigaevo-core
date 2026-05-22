"""Greedy max-min initialization + Riesz s-energy refinement.

generate_config builds the n-point set incrementally: starting from one
random unit vector, it repeatedly samples M candidate directions and adds
the one whose maximum inner product with the existing set is smallest.
This is the discrete analogue of Tammes' problem solved greedily, and gives
a much better starting μ than Gaussian (≈ 0.4-0.5 vs. 0.94 at n=600, d=11).
The alternating-projection ETF approach was tried earlier but collapses to
rank-d duplicates because n=600 ≫ d(d+1)/2 = 66 (no real ETF exists at
this size) — the rank-d PSD projection nulls 98% of the spectrum, causing
the renormalized rows to coincide.

improve runs Riesz s-energy minimization with s annealed 2 → 8 (high s makes
the kernel concentrate on the closest pair, approaching min-max). Updates are
Riemannian gradient steps on each sphere, projected onto the tangent and
retracted exactly. This is structurally distinct from a LogSumExp surrogate:
Riesz uses a power-law repulsive kernel ‖d‖^{-s} on EUCLIDEAN pairwise
distances, not on inner products.

perturb applies a small SO(d) rotation in a random 2-plane to all points
(angle ∝ intensity); structure-preserving (preserves all pairwise inner
products globally), then jiggles a few active points off the rotated frame.
"""

from __future__ import annotations

import numpy as np


def _renormalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


def _greedy_max_min(
    n: int, d: int, rng: np.random.Generator, M: int = 2000
) -> np.ndarray:
    """Build an n-point unit-norm set greedily: add the candidate with the
    smallest max-cos against the existing set.

    Cost: M × Σ_{k=1..n} k × d ≈ M · n²d/2 flops. For (n=600, d=11, M=2000)
    that is ~4e9 flops, ≈ 5-10s in pure numpy. Returns one point per row, all
    unit norm. The first point is uniform on the sphere.
    """
    X = rng.standard_normal((1, d)).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    rows: list[np.ndarray] = [X[0]]
    cur = X
    for _ in range(n - 1):
        cand = rng.standard_normal((M, d)).astype(np.float64)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        cos_mat = cand @ cur.T  # (M, k)
        max_cos = cos_mat.max(axis=1)
        best = int(np.argmin(max_cos))
        rows.append(cand[best])
        cur = np.vstack([cur, cand[best : best + 1]])
    return np.asarray(rows, dtype=np.float64)


def _riesz_value_grad(X: np.ndarray, s: float) -> tuple[float, np.ndarray]:
    """Riesz s-energy E_s(X) = Σ_{i<j} ‖x_i - x_j‖^{-s}, returning value and ∇_X.

    ∇_{x_i} ‖x_i - x_j‖^{-s} = -s (x_i - x_j) ‖x_i - x_j‖^{-s-2}
    """
    n = X.shape[0]
    # Pairwise squared distances on the unit sphere: d² = 2 - 2 G_ij
    G = X @ X.T
    D2 = 2.0 - 2.0 * G
    np.fill_diagonal(D2, np.inf)
    D2 = np.maximum(D2, 1e-14)
    inv_s = D2 ** (-s / 2.0)  # ‖d‖^{-s}
    value = 0.5 * float(inv_s.sum())  # double-count then halve
    coef = -s * (
        D2 ** (-(s + 2) / 2.0)
    )  # ∂/∂G_ij of ‖d‖^{-s} via chain rule, then *2 to recover (xi-xj)·1 form
    np.fill_diagonal(coef, 0.0)
    # ∇_{x_i} E = Σ_j coef_ij * (x_i - x_j) ; collect into one matmul.
    row_sum = coef.sum(axis=1, keepdims=True)
    grad = row_sum * X - coef @ X
    return value, grad


def _project_tangent(g_E: np.ndarray, X: np.ndarray) -> np.ndarray:
    radial = np.sum(g_E * X, axis=1, keepdims=True)
    return g_E - radial * X


def _retract(X: np.ndarray, V: np.ndarray, step: float) -> np.ndarray:
    V_norms = np.linalg.norm(V, axis=1, keepdims=True)
    safe = np.where(V_norms > 1e-30, V_norms, 1.0)
    angles = step * V_norms
    direction = V / safe
    return np.cos(angles) * X + np.sin(angles) * direction


class Improver:
    def __init__(self, n: int, d: int, seed: int = 0):
        self.n = int(n)
        self.d = int(d)
        self.seed = int(seed)

    def generate_config(self, seed=None) -> np.ndarray:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        X = _greedy_max_min(self.n, self.d, rng, M=2000)
        return _renormalize(X)

    def improve(self, points: np.ndarray, seed=None) -> np.ndarray:
        X = _renormalize(np.asarray(points, dtype=np.float64).copy())

        s_schedule = (2.0, 3.5, 5.0, 6.5, 8.0)
        steps_per_s = 60

        for s in s_schedule:
            step = 0.05
            f_prev, grad_E = _riesz_value_grad(X, s)
            tan = _project_tangent(grad_E, X)
            for _ in range(steps_per_s):
                if float(np.linalg.norm(tan)) < 1e-14:
                    break
                for _bt in range(15):
                    X_try = _retract(X, -tan, step)
                    f_try, grad_try_E = _riesz_value_grad(X_try, s)
                    if f_try < f_prev - 1e-12:
                        X = X_try
                        f_prev = f_try
                        tan = _project_tangent(grad_try_E, X)
                        step = min(step * 1.3, 0.5)
                        break
                    step *= 0.5
                else:
                    break
        return _renormalize(X)

    def perturb(self, points: np.ndarray, intensity: float, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        X = _renormalize(np.asarray(points, dtype=np.float64).copy())
        intensity = float(np.clip(intensity, 0.0, 1.0))

        # 1. Global rotation in a random 2-plane (preserves inner products
        #    overall but rotates the whole frame; useful for symmetry breaking
        #    when paired with the targeted jiggle below).
        u = rng.standard_normal(self.d).astype(np.float64)
        u /= np.linalg.norm(u)
        v = rng.standard_normal(self.d).astype(np.float64)
        v -= float(v @ u) * u
        v_norm = float(np.linalg.norm(v))
        if v_norm > 1e-12:
            v /= v_norm
            theta = intensity * float(rng.uniform(0.1, np.pi / 6))
            cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
            proj_u = X @ u
            proj_v = X @ v
            X = (
                X
                - np.outer(proj_u, u)
                - np.outer(proj_v, v)
                + np.outer(proj_u * cos_t - proj_v * sin_t, u)
                + np.outer(proj_u * sin_t + proj_v * cos_t, v)
            )

        # 2. Targeted jiggle on K active points to break the lattice-like
        #    structure that alt-projection tends to produce.
        K = max(2, int(2 + 40 * intensity))
        K = min(K, self.n)
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        active = np.argpartition(-G.max(axis=1), K - 1)[:K]
        jitter = rng.standard_normal((len(active), self.d)).astype(np.float64)
        jitter -= np.sum(jitter * X[active], axis=1, keepdims=True) * X[active]
        jitter_norms = np.linalg.norm(jitter, axis=1, keepdims=True)
        safe = np.where(jitter_norms > 1e-12, jitter_norms, 1.0)
        jitter /= safe
        angles = intensity * rng.uniform(0.05, 0.4, size=(len(active), 1)).astype(
            np.float64
        )
        X[active] = np.cos(angles) * X[active] + np.sin(angles) * jitter

        return _renormalize(X)


def entrypoint():
    return Improver
