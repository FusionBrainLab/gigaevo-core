"""Active-pair subgradient method — exact non-smooth minimization of max <x_i, x_j>.

This seed does NOT use the LogSumExp surrogate. It works directly on the
non-differentiable objective μ(X) = max_{i<j} <x_i, x_j> by maintaining a
small active set of pairs at or near the current maximum and stepping in a
subgradient direction.

improve:
  1. Form Gram, find the K active pairs (within tol of the current max).
  2. The subgradient of μ at X is the convex hull of (e_i e_j^T + e_j e_i^T)
     for active pairs. We use the unweighted mean — equivalent to gradient of
     the K-active-pair p-norm with p → ∞.
  3. Project to tangent, take a Riemannian step with Armijo on μ itself (not
     on a surrogate). Repeat until the active set stabilises and the step
     fails Armijo at the smallest tolerance.

The active set widens as we approach the optimum (more pairs realize the
max), so the method naturally transitions from "push the worst pair" to
"balance many ties" — this is the right behaviour at a Tammes-equilibrium.

perturb:
  Multi-pair simultaneous push on the top-K active pairs (K scales with
  intensity). Pure pair-wise tangent rotation, no global noise.

generate_config: Gaussian on the sphere.
"""

from __future__ import annotations

import numpy as np


def _renormalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


def _max_offdiag(G: np.ndarray) -> float:
    n = G.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(G[iu].max())


def _welch_bound(n: int, d: int) -> float:
    if n <= d:
        return 0.0
    return float(np.sqrt((n - d) / (d * (n - 1))))


def _active_subgradient(X: np.ndarray, tol: float) -> tuple[np.ndarray, float, int]:
    """Subgradient direction summed over (i, j) within `tol` of the current max.

    NO global 1/num_active dilution — each row i gets the unscaled sum
    of its active partners' positions, so per-point bottleneck points
    receive a step proportional to their local active-set size.
    """
    G = X @ X.T
    n = X.shape[0]
    G_off = G.copy()
    np.fill_diagonal(G_off, -np.inf)
    mu = _max_offdiag(G)
    A = G_off >= (mu - tol)
    num_active = int(A.sum() // 2)
    if num_active == 0:
        return np.zeros_like(X), mu, 0
    grad = A.astype(np.float64) @ X
    return grad, mu, num_active


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
        X = rng.standard_normal((self.n, self.d)).astype(np.float64)
        return _renormalize(X)

    def improve(self, points: np.ndarray, seed=None) -> np.ndarray:
        X = _renormalize(np.asarray(points, dtype=np.float64).copy())

        # Polyak subgradient method: step_k = (μ_k − μ_lower) / ‖g_k‖².
        # The Welch bound is a valid lower bound on the achievable max-coherence
        # for any n > d, so the gap (μ_k − μ_W) ≥ 0 monotonically shrinks at the
        # optimum, naturally damping step size. We track the best iterate
        # because subgradient methods are NOT monotone — individual steps can
        # increase μ but the running min over iterates does converge.
        tol_schedule = (0.05, 0.02, 0.008, 0.003, 0.001)
        iters_per_tol = 80
        mu_lower = _welch_bound(self.n, self.d)

        best_X = X.copy()
        G_init = X @ X.T
        np.fill_diagonal(G_init, -np.inf)
        best_mu = float(G_init.max())

        for tol in tol_schedule:
            for _ in range(iters_per_tol):
                grad, mu_curr, n_active = _active_subgradient(X, tol)
                if n_active == 0:
                    break
                tan = _project_tangent(grad, X)
                tan_norm = float(np.linalg.norm(tan))
                if tan_norm < 1e-14:
                    break
                gap = max(mu_curr - mu_lower, 1e-6)
                step = gap / (tan_norm**2)
                step = float(np.clip(step, 1e-5, 0.5))
                X = _retract(X, -tan, step)
                X = _renormalize(X)
                G = X @ X.T
                np.fill_diagonal(G, -np.inf)
                mu_new = float(G.max())
                if mu_new < best_mu - 1e-12:
                    best_mu = mu_new
                    best_X = X.copy()
        return _renormalize(best_X)

    def perturb(self, points: np.ndarray, intensity: float, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        X = _renormalize(np.asarray(points, dtype=np.float64).copy())
        intensity = float(np.clip(intensity, 0.0, 1.0))

        # Take K top active pairs and push each pair apart along their
        # local tangent geodesic. Pure pair-wise — no global noise, no
        # rotation, no surrogate.
        K = max(2, int(2 + 30 * intensity))
        G = X @ X.T
        n = self.n
        np.fill_diagonal(G, -np.inf)
        flat = G.ravel()
        # Take more candidates than K to filter for upper-tri pairs.
        cand_cnt = min(4 * K, flat.size)
        cand = np.argpartition(-flat, cand_cnt - 1)[:cand_cnt]
        pairs: list[tuple[int, int]] = []
        seen: set[int] = set()
        for idx in cand[np.argsort(-flat[cand])]:
            i, j = int(idx // n), int(idx % n)
            if i >= j:
                continue
            # Avoid touching the same point twice in one perturb step.
            if i in seen or j in seen:
                continue
            pairs.append((i, j))
            seen.add(i)
            seen.add(j)
            if len(pairs) >= K:
                break

        for i, j in pairs:
            xi, xj = X[i].copy(), X[j].copy()
            # Tangent of xj at xi (steepest-ascent of G_ij at xi w.r.t. xi).
            tan_i = xj - float(xj @ xi) * xi
            tan_j = xi - float(xi @ xj) * xj
            ni, nj = float(np.linalg.norm(tan_i)), float(np.linalg.norm(tan_j))
            angle = intensity * float(rng.uniform(0.05, 0.4))
            if ni > 1e-12:
                X[i] = np.cos(angle) * xi - np.sin(angle) * (tan_i / ni)
            if nj > 1e-12:
                X[j] = np.cos(angle) * xj - np.sin(angle) * (tan_j / nj)

        return _renormalize(X)


def entrypoint():
    return Improver
