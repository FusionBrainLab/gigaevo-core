"""Helper functions for the adapted AlgoTune ``feedback_controller_design`` task.

This mirrors the core problem generation and validation logic from
``AlgoTune-main/AlgoTuneTasks/feedback_controller_design/feedback_controller_design.py``
without importing the full AlgoTune runtime or relying on ``cvxpy``.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from scipy.linalg import solve as linalg_solve
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

STABILITY_TOLERANCE = 1.0e-10
POSITIVE_DEFINITE_TOLERANCE = 1.0e-10
NEGATIVE_DEFINITE_TOLERANCE = 1.0e-10
SYMMETRY_RTOL = 1.0e-5
SYMMETRY_ATOL = 1.0e-8
GEOMETRIC_SERIES_TOLERANCE = 1.0e-12
GEOMETRIC_SERIES_MAX_ITERATIONS = 10_000
RANK_SAFETY_FACTOR = 100.0


class CaseSpec(TypedDict):
    n: int
    random_seed: int


def get_case_specs() -> list[CaseSpec]:
    """Deterministic evaluation suite spanning small and medium systems."""
    return [
        {"n": 2, "random_seed": 1234},
        {"n": 3, "random_seed": 2024},
        {"n": 4, "random_seed": 7},
        {"n": 6, "random_seed": 31415},
        {"n": 8, "random_seed": 2718},
    ]


def generate_problem(n: int = 2, random_seed: int = 1234) -> dict[str, list[list[float]]]:
    """Generate one controller-design instance using AlgoTune's logic."""
    n = max(n, 2)
    rng = np.random.default_rng(random_seed)
    m = int(n // 2)

    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, m))

    return {"A": A.tolist(), "B": B.tolist()}


def _as_matrix(problem: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(problem[key], dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{key} must be a 2D matrix, got shape {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain only finite values.")
    return value


def _parse_problem(problem: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    required = {"A", "B"}
    missing = required.difference(problem)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"Problem dictionary is missing keys: {missing_keys}.")

    A = _as_matrix(problem, "A")
    B = _as_matrix(problem, "B")
    n = A.shape[0]

    if A.shape != (n, n):
        raise ValueError(f"A must be square, got shape {A.shape}.")
    if B.shape[0] != n:
        raise ValueError("B must have the same number of rows as A.")
    if B.shape[1] <= 0:
        raise ValueError("B must have at least one input column.")

    return A, B


def spectral_radius(A: np.ndarray) -> float:
    """Return the spectral radius of ``A``."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def is_stable_matrix(A: np.ndarray) -> bool:
    """Check discrete-time asymptotic stability via the spectral radius."""
    return spectral_radius(A) < 1.0 - STABILITY_TOLERANCE


def _matrix_rank(M: np.ndarray) -> int:
    singular_values = np.linalg.svd(M, compute_uv=False)
    if singular_values.size == 0:
        return 0
    tolerance = (
        max(M.shape)
        * np.finfo(np.float64).eps
        * max(1.0, float(singular_values[0]))
        * RANK_SAFETY_FACTOR
    )
    return int(np.sum(singular_values > tolerance))


def is_stabilizable_pair(A: np.ndarray, B: np.ndarray) -> bool:
    """Check stabilizability with the PBH rank test on unstable modes."""
    n = A.shape[0]
    A_complex = np.asarray(A, dtype=np.complex128)
    B_complex = np.asarray(B, dtype=np.complex128)

    for eigenvalue in np.linalg.eigvals(A_complex):
        if abs(eigenvalue) < 1.0 - STABILITY_TOLERANCE:
            continue
        pbh_matrix = np.hstack((eigenvalue * np.eye(n, dtype=np.complex128) - A_complex, B_complex))
        if _matrix_rank(pbh_matrix) < n:
            return False
    return True


def _solve_lyapunov_certificate(A: np.ndarray) -> np.ndarray:
    """Solve ``A.T @ P @ A - P = -I`` for stable closed-loop systems."""
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


def _compute_stabilizing_gain(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, m = B.shape
    if is_stable_matrix(A):
        K = np.zeros((m, n), dtype=np.float64)
        return K, _solve_lyapunov_certificate(A)

    Q = np.eye(n, dtype=np.float64)
    R = np.eye(m, dtype=np.float64)

    try:
        try:
            X = solve_discrete_are(A, B, Q, R, balanced=False)
        except TypeError:
            X = solve_discrete_are(A, B, Q, R)
    except Exception as exc:
        raise RuntimeError("Failed to solve the discrete algebraic Riccati equation.") from exc

    X = np.asarray(np.real_if_close(X, tol=1000), dtype=np.float64)
    X = 0.5 * (X + X.T)

    gain_lhs = R + B.T @ X @ B
    gain_rhs = B.T @ X @ A
    try:
        K = -linalg_solve(gain_lhs, gain_rhs, assume_a="pos")
    except np.linalg.LinAlgError:
        K = -np.linalg.pinv(gain_lhs) @ gain_rhs

    K = np.asarray(np.real_if_close(K, tol=1000), dtype=np.float64)
    closed_loop = A + B @ K
    if not is_stable_matrix(closed_loop):
        raise RuntimeError("Computed feedback gain is not stabilizing.")

    return K, _solve_lyapunov_certificate(closed_loop)


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Reference solver mirroring the controller-design task semantics."""
    A, B = _parse_problem(problem)
    if not is_stabilizable_pair(A, B):
        return {"is_stabilizable": False, "K": None, "P": None}

    K, P = _compute_stabilizing_gain(A, B)
    return {"is_stabilizable": True, "K": K.tolist(), "P": P.tolist()}


def validate_solution(problem: dict[str, Any], solution: Any) -> dict[str, float]:
    """Validate one candidate controller and return diagnostic margins."""
    if not isinstance(solution, dict):
        raise TypeError(f"Expected dict solution, got {type(solution).__name__}.")
    if not all(key in solution for key in ("is_stabilizable", "K", "P")):
        raise ValueError("Solution must contain 'is_stabilizable', 'K', and 'P'.")

    raw_is_stabilizable = solution["is_stabilizable"]
    if not isinstance(raw_is_stabilizable, (bool, np.bool_)):
        raise TypeError("'is_stabilizable' must be a boolean.")
    is_stabilizable = bool(raw_is_stabilizable)

    A, B = _parse_problem(problem)
    n, m = B.shape
    true_is_stabilizable = is_stabilizable_pair(A, B)
    if is_stabilizable != true_is_stabilizable:
        raise ValueError(
            "Incorrect stabilizability classification: "
            f"expected {true_is_stabilizable}, got {is_stabilizable}."
        )

    if not is_stabilizable:
        if solution["K"] is not None or solution["P"] is not None:
            raise ValueError("K and P must be None for non-stabilizable systems.")
        return {
            "closed_loop_spectral_radius": 0.0,
            "min_p_eigenvalue": 0.0,
            "max_decay_eigenvalue": 0.0,
        }

    if solution["K"] is None:
        raise ValueError("Stabilizable systems must provide a feedback gain K.")
    if solution["P"] is None:
        raise ValueError("Stabilizable systems must provide a Lyapunov matrix P.")

    K = np.asarray(solution["K"], dtype=np.float64)
    P = np.asarray(solution["P"], dtype=np.float64)

    if K.shape != (m, n):
        raise ValueError(f"K has wrong shape: expected {(m, n)}, got {K.shape}.")
    if P.shape != (n, n):
        raise ValueError(f"P has wrong shape: expected {(n, n)}, got {P.shape}.")
    if not np.all(np.isfinite(K)):
        raise ValueError("K must contain only finite values.")
    if not np.all(np.isfinite(P)):
        raise ValueError("P must contain only finite values.")
    if not np.allclose(P, P.T, rtol=SYMMETRY_RTOL, atol=SYMMETRY_ATOL):
        raise ValueError("P is not symmetric.")

    P = 0.5 * (P + P.T)
    closed_loop = A + B @ K
    closed_loop_spectral_radius = spectral_radius(closed_loop)
    if not is_stable_matrix(closed_loop):
        raise ValueError(
            "Closed-loop system is not asymptotically stable: "
            f"spectral radius={closed_loop_spectral_radius}."
        )

    min_p_eigenvalue = float(np.min(np.linalg.eigvalsh(P)))
    if min_p_eigenvalue < POSITIVE_DEFINITE_TOLERANCE:
        raise ValueError(
            "P is not positive definite enough to certify asymptotic stability."
        )

    decay_matrix = closed_loop.T @ P @ closed_loop - P
    decay_matrix = 0.5 * (decay_matrix + decay_matrix.T)
    max_decay_eigenvalue = float(np.max(np.linalg.eigvalsh(decay_matrix)))
    if max_decay_eigenvalue > NEGATIVE_DEFINITE_TOLERANCE:
        raise ValueError("Closed-loop Lyapunov inequality is not negative definite.")

    return {
        "closed_loop_spectral_radius": closed_loop_spectral_radius,
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
