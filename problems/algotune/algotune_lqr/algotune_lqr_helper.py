"""Helper functions for the adapted AlgoTune ``lqr`` task.

This mirrors the core problem generation and validation logic from
``AlgoTune-main/AlgoTuneTasks/lqr/lqr.py`` without importing the full
AlgoTune runtime.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import numpy.testing as npt
from scipy.linalg import solve as linalg_solve

COST_RTOL = 1.0e-5
COST_ATOL = 1.0e-7


class CaseSpec(TypedDict):
    n: int
    random_seed: int


def get_case_specs() -> list[CaseSpec]:
    """Deterministic evaluation suite spanning small and medium LQR instances."""
    return [
        {"n": 1, "random_seed": 1234},
        {"n": 2, "random_seed": 2024},
        {"n": 3, "random_seed": 7},
        {"n": 6, "random_seed": 31415},
        {"n": 8, "random_seed": 2718},
    ]


def generate_problem(n: int = 1, random_seed: int = 1234) -> dict[str, Any]:
    """Generate one finite-horizon LQR instance using AlgoTune's logic."""
    rng = np.random.default_rng(random_seed)
    m = max(1, n // 2)
    horizon = 2 * n

    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, m))

    Q = rng.standard_normal((n, n))
    Q = Q.T @ Q
    P = rng.standard_normal((n, n))
    P = P.T @ P

    R = rng.standard_normal((m, m))
    R = R.T @ R + 1.0e-2 * np.eye(m)

    x0 = rng.standard_normal((n, 1))

    return {
        "A": A.tolist(),
        "B": B.tolist(),
        "Q": Q.tolist(),
        "R": R.tolist(),
        "P": P.tolist(),
        "T": int(horizon),
        "x0": x0.tolist(),
    }


def _as_matrix(problem: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(problem[key], dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{key} must be a 2D matrix, got shape {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain only finite values.")
    return value


def _as_column(problem: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(problem[key], dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"{key} must be a column vector, got shape {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain only finite values.")
    return value


def _parse_problem(problem: dict[str, Any]) -> tuple[np.ndarray, ...]:
    required = {"A", "B", "Q", "R", "P", "T", "x0"}
    missing = required.difference(problem)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"Problem dictionary is missing keys: {missing_keys}.")

    A = _as_matrix(problem, "A")
    B = _as_matrix(problem, "B")
    Q = _as_matrix(problem, "Q")
    R = _as_matrix(problem, "R")
    P = _as_matrix(problem, "P")
    x0 = _as_column(problem, "x0")
    horizon = int(problem["T"])

    n = A.shape[0]
    m = B.shape[1]
    if A.shape != (n, n):
        raise ValueError(f"A must be square, got shape {A.shape}.")
    if B.shape[0] != n:
        raise ValueError("B must have the same number of rows as A.")
    if Q.shape != (n, n):
        raise ValueError("Q must match the state dimension.")
    if P.shape != (n, n):
        raise ValueError("P must match the state dimension.")
    if R.shape != (m, m):
        raise ValueError("R must match the control dimension.")
    if x0.shape != (n, 1):
        raise ValueError("x0 must match the state dimension.")
    if horizon <= 0:
        raise ValueError("T must be a positive integer.")

    return A, B, Q, R, P, horizon, x0


def _backward_riccati_gains(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Riccati cost-to-go matrices and feedback gains."""
    n, m = B.shape
    S = np.zeros((horizon + 1, n, n), dtype=np.float64)
    K = np.zeros((horizon, m, n), dtype=np.float64)
    S[horizon] = P

    for t in range(horizon - 1, -1, -1):
        next_s = S[t + 1]
        lhs = R + B.T @ next_s @ B
        rhs = B.T @ next_s @ A
        try:
            K[t] = linalg_solve(lhs, rhs, assume_a="pos")
        except np.linalg.LinAlgError:
            K[t] = np.linalg.pinv(lhs) @ rhs
        closed_loop = A - B @ K[t]
        S[t] = Q + K[t].T @ R @ K[t] + closed_loop.T @ next_s @ closed_loop
        S[t] = 0.5 * (S[t] + S[t].T)

    return S, K


def solve_problem(problem: dict[str, Any]) -> dict[str, list[list[float]]]:
    """Reference solver mirroring AlgoTune's LQR task semantics."""
    A, B, Q, R, P, horizon, x0 = _parse_problem(problem)
    _, gains = _backward_riccati_gains(A, B, Q, R, P, horizon)

    _, m = B.shape
    U = np.zeros((horizon, m), dtype=np.float64)
    x = x0.copy()
    for t in range(horizon):
        u = -gains[t] @ x
        U[t] = u.ravel()
        x = A @ x + B @ u

    return {"U": U.tolist()}


def compute_cost(problem: dict[str, Any], U: np.ndarray) -> float:
    """Simulate the LQR dynamics and return the total finite-horizon cost."""
    A, B, Q, R, P, horizon, x0 = _parse_problem(problem)
    n, m = B.shape
    if U.shape != (horizon, m):
        raise ValueError(f"U has wrong shape: expected {(horizon, m)}, got {U.shape}.")
    if not np.all(np.isfinite(U)):
        raise ValueError("U must contain only finite values.")

    X = np.zeros((horizon + 1, n, 1), dtype=np.float64)
    X[0] = x0
    for t in range(horizon):
        u = U[t].reshape(m, 1)
        X[t + 1] = A @ X[t] + B @ u

    if not np.all(np.isfinite(X)):
        raise ValueError("State trajectory contains non-finite values.")

    total_cost = 0.0
    for t in range(horizon):
        xt = X[t]
        ut = U[t].reshape(m, 1)
        total_cost += float(xt.T @ Q @ xt + ut.T @ R @ ut)
    total_cost += float(X[horizon].T @ P @ X[horizon])
    return total_cost


def optimal_cost(problem: dict[str, Any]) -> float:
    """Return the optimal LQR cost via backward Riccati recursion."""
    A, B, Q, R, P, horizon, x0 = _parse_problem(problem)
    S, _ = _backward_riccati_gains(A, B, Q, R, P, horizon)
    return float(x0.T @ S[0] @ x0)


def relative_cost_error(reference_cost: float, candidate_cost: float) -> float:
    """Return the scale-aware relative cost error."""
    scale = max(1.0, abs(reference_cost))
    return float(abs(candidate_cost - reference_cost) / scale)


def validate_solution(problem: dict[str, Any], solution: Any) -> dict[str, float]:
    """Validate one candidate LQR solution and return diagnostic costs."""
    if not isinstance(solution, dict):
        raise TypeError(f"Expected dict solution, got {type(solution).__name__}.")
    if "U" not in solution:
        raise ValueError("Solution must contain key 'U'.")

    U = np.asarray(solution["U"], dtype=np.float64)
    candidate_cost = compute_cost(problem, U)
    reference_cost = optimal_cost(problem)

    try:
        npt.assert_allclose(
            candidate_cost, reference_cost, rtol=COST_RTOL, atol=COST_ATOL
        )
    except AssertionError as exc:
        raise ValueError(
            f"LQR cost mismatch: candidate={candidate_cost}, optimal={reference_cost}."
        ) from exc

    return {
        "candidate_cost": candidate_cost,
        "optimal_cost": reference_cost,
        "relative_cost_error": relative_cost_error(reference_cost, candidate_cost),
    }


def is_solution(problem: dict[str, Any], solution: Any) -> bool:
    """Return whether ``solution`` satisfies the adapted validator."""
    try:
        validate_solution(problem, solution)
    except Exception:
        return False
    return True
