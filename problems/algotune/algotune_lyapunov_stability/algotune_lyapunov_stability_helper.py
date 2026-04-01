"""Helper functions for the adapted AlgoTune ``lyapunov_stability`` task.

This mirrors the core problem generation and validation logic from
``AlgoTune-main/AlgoTuneTasks/lyapunov_stability/lyapunov_stability.py``
without importing the full AlgoTune runtime or relying on ``cvxpy``, which is
not installed in this environment.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

STABILITY_TOLERANCE = 1.0e-10
POSITIVE_DEFINITE_TOLERANCE = 1.0e-10
NEGATIVE_DEFINITE_TOLERANCE = 1.0e-10
SYMMETRY_RTOL = 1.0e-5
SYMMETRY_ATOL = 1.0e-8
GEOMETRIC_SERIES_TOLERANCE = 1.0e-12
GEOMETRIC_SERIES_MAX_ITERATIONS = 10_000


class CaseSpec(TypedDict):
    n: int
    random_seed: int


def get_case_specs() -> list[CaseSpec]:
    """Deterministic evaluation suite spanning stable and unstable systems."""
    return [
        {"n": 1, "random_seed": 1234},
        {"n": 2, "random_seed": 2024},
        {"n": 3, "random_seed": 7},
        {"n": 6, "random_seed": 31415},
        {"n": 8, "random_seed": 2718},
    ]


def generate_problem(
    n: int = 1, random_seed: int = 1234
) -> dict[str, list[list[float]]]:
    """Generate one Lyapunov stability instance using AlgoTune's logic."""
    rng = np.random.default_rng(random_seed)
    A = rng.normal(0, 1, (n, n))

    make_stable = rng.random() < 0.7
    if make_stable:
        eigenvalues = np.linalg.eigvals(A)
        max_magnitude = float(np.max(np.abs(eigenvalues)))
        scaling_factor = 0.8 / max_magnitude
        A = A * scaling_factor

    return {"A": A.tolist()}


def _as_matrix(problem: dict[str, Any]) -> np.ndarray:
    A = np.asarray(problem["A"], dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("System matrix A must contain only finite values.")
    return A


def spectral_radius(A: np.ndarray) -> float:
    """Return the spectral radius of ``A``."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def is_stable_matrix(A: np.ndarray) -> bool:
    """Check discrete-time asymptotic stability via the spectral radius."""
    return spectral_radius(A) < 1.0 - STABILITY_TOLERANCE


def _solve_lyapunov_certificate(A: np.ndarray) -> np.ndarray:
    """Solve ``A.T @ P @ A - P = -I`` for stable systems."""
    n = A.shape[0]
    identity = np.eye(n, dtype=np.float64)

    try:
        P = solve_discrete_lyapunov(A.T, identity)
        return 0.5 * (P + P.T)
    except Exception:
        pass

    P = identity.copy()
    term = identity.copy()
    for _ in range(GEOMETRIC_SERIES_MAX_ITERATIONS):
        term = A.T @ term @ A
        P = P + term
        if np.linalg.norm(term, ord="fro") <= GEOMETRIC_SERIES_TOLERANCE * (
            1.0 + np.linalg.norm(P, ord="fro")
        ):
            return 0.5 * (P + P.T)

    raise RuntimeError("Failed to compute a Lyapunov certificate for a stable system.")


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Reference solver mirroring the Lyapunov stability task semantics."""
    A = _as_matrix(problem)
    if not is_stable_matrix(A):
        return {"is_stable": False, "P": None}

    P = _solve_lyapunov_certificate(A)
    return {"is_stable": True, "P": P.tolist()}


def validate_solution(problem: dict[str, Any], solution: Any) -> dict[str, float]:
    """Validate one candidate solution and return diagnostic margins."""
    if not isinstance(solution, dict):
        raise TypeError(f"Expected dict solution, got {type(solution).__name__}.")
    if "is_stable" not in solution or "P" not in solution:
        raise ValueError("Solution must contain 'is_stable' and 'P'.")

    raw_is_stable = solution["is_stable"]
    if not isinstance(raw_is_stable, (bool, np.bool_)):
        raise TypeError("'is_stable' must be a boolean.")
    is_stable = bool(raw_is_stable)

    A = _as_matrix(problem)
    reference = solve_problem(problem)
    true_is_stable = bool(reference["is_stable"])
    if is_stable != true_is_stable:
        raise ValueError(
            f"Incorrect stability classification: expected {true_is_stable}, got {is_stable}."
        )

    if not is_stable:
        return {"min_p_eigenvalue": 0.0, "max_decay_eigenvalue": 0.0}

    if solution["P"] is None:
        raise ValueError("Stable systems must provide a Lyapunov matrix P.")

    P = np.asarray(solution["P"], dtype=np.float64)
    if P.shape != A.shape:
        raise ValueError(f"P has wrong shape: expected {A.shape}, got {P.shape}.")
    if not np.all(np.isfinite(P)):
        raise ValueError("P must contain only finite values.")
    if not np.allclose(P, P.T, rtol=SYMMETRY_RTOL, atol=SYMMETRY_ATOL):
        raise ValueError("P is not symmetric.")

    P = 0.5 * (P + P.T)
    min_p_eigenvalue = float(np.min(np.linalg.eigvalsh(P)))
    if min_p_eigenvalue < POSITIVE_DEFINITE_TOLERANCE:
        raise ValueError(
            "P is not positive definite enough to certify asymptotic stability."
        )

    decay_matrix = 0.5 * ((A.T @ P @ A - P) + (A.T @ P @ A - P).T)
    max_decay_eigenvalue = float(np.max(np.linalg.eigvalsh(decay_matrix)))
    if max_decay_eigenvalue > NEGATIVE_DEFINITE_TOLERANCE:
        raise ValueError("A^T P A - P is not negative definite.")

    return {
        "min_p_eigenvalue": min_p_eigenvalue,
        "max_decay_eigenvalue": max_decay_eigenvalue,
    }


def is_solution(problem: dict[str, Any], solution: Any) -> bool:
    """Return whether ``solution`` satisfies the adapted validator."""
    try:
        validate_solution(problem, solution)
    except Exception:
        return False
    return True
