"""Helper functions for the adapted AlgoTune ``kalman_filter`` task.

This mirrors the core problem generation and validation logic from
``AlgoTune-main/AlgoTuneTasks/kalman_filter/kalman_filter.py`` without
importing the full AlgoTune runtime or relying on ``cvxpy``.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

FEASIBILITY_TOLERANCE = 1.0e-5
OBJECTIVE_RTOL = 1.0e-5


class CaseSpec(TypedDict):
    n: int
    random_seed: int


def get_case_specs() -> list[CaseSpec]:
    """Deterministic evaluation suite spanning small and medium instances."""
    return [
        {"n": 1, "random_seed": 1234},
        {"n": 2, "random_seed": 2024},
        {"n": 3, "random_seed": 7},
        {"n": 6, "random_seed": 31415},
        {"n": 8, "random_seed": 2718},
    ]


def generate_problem(n: int = 1, random_seed: int = 1234) -> dict[str, Any]:
    """Generate one Kalman-filter instance using AlgoTune's logic."""
    rng = np.random.default_rng(random_seed)
    horizon = 2 * n
    p = n
    m = n
    tau = 1.0

    A_raw = rng.standard_normal((n, n))
    eigs = np.linalg.eigvals(A_raw)
    A = A_raw / (np.max(np.abs(eigs)) + 1.0e-3)
    B = 0.5 * rng.standard_normal((n, p))
    C = rng.standard_normal((m, n))

    x0 = rng.standard_normal(n)
    x_true = np.zeros((horizon + 1, n), dtype=np.float64)
    x_true[0] = x0
    w_true = 0.1 * rng.standard_normal((horizon, p))
    v_true = 0.1 * rng.standard_normal((horizon, m))
    y = np.zeros((horizon, m), dtype=np.float64)
    for t in range(horizon):
        x_true[t + 1] = A @ x_true[t] + B @ w_true[t]
        y[t] = C @ x_true[t] + v_true[t]

    return {
        "A": A.tolist(),
        "B": B.tolist(),
        "C": C.tolist(),
        "y": y.tolist(),
        "x_initial": x0.tolist(),
        "tau": tau,
    }


def _as_matrix(problem: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(problem[key], dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{key} must be a 2D matrix, got shape {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain only finite values.")
    return value


def _as_vector(problem: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(problem[key], dtype=np.float64)
    if value.ndim != 1:
        raise ValueError(f"{key} must be a 1D vector, got shape {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain only finite values.")
    return value


def _parse_problem(problem: dict[str, Any]) -> tuple[np.ndarray, ...]:
    required = {"A", "B", "C", "y", "x_initial", "tau"}
    missing = required.difference(problem)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"Problem dictionary is missing keys: {missing_keys}.")

    A = _as_matrix(problem, "A")
    B = _as_matrix(problem, "B")
    C = _as_matrix(problem, "C")
    y = _as_matrix(problem, "y")
    x_initial = _as_vector(problem, "x_initial")
    tau = float(problem["tau"])

    n = A.shape[0]
    p = B.shape[1]
    m = C.shape[0]
    horizon = y.shape[0]

    if A.shape != (n, n):
        raise ValueError(f"A must be square, got shape {A.shape}.")
    if B.shape[0] != n:
        raise ValueError("B must have the same number of rows as A.")
    if C.shape[1] != n:
        raise ValueError("C must have the same number of columns as A.")
    if y.shape[1] != m:
        raise ValueError("y must have the same number of columns as C.")
    if x_initial.shape != (n,):
        raise ValueError("x_initial must match the state dimension.")
    if horizon <= 0:
        raise ValueError("y must contain at least one time step.")
    if p <= 0 or m <= 0:
        raise ValueError("B and C must define positive process and measurement dimensions.")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be a finite positive scalar.")

    return A, B, C, y, x_initial, tau


def _build_measurement_system(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    x_initial: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the affine measurement model ``y = d + M w + v``."""
    n = A.shape[0]
    p = B.shape[1]
    m = C.shape[0]

    powers: list[np.ndarray] = [np.eye(n, dtype=np.float64)]
    for _ in range(horizon):
        powers.append(powers[-1] @ A)

    baseline = np.zeros((horizon, m), dtype=np.float64)
    measurement_matrix = np.zeros((horizon * m, horizon * p), dtype=np.float64)

    for t in range(horizon):
        baseline[t] = C @ (powers[t] @ x_initial)
        row_slice = slice(t * m, (t + 1) * m)
        for k in range(t):
            col_slice = slice(k * p, (k + 1) * p)
            measurement_matrix[row_slice, col_slice] = C @ (powers[t - 1 - k] @ B)

    return baseline.reshape(-1), measurement_matrix


def _solve_process_noise(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    y: np.ndarray,
    x_initial: np.ndarray,
    tau: float,
) -> np.ndarray:
    horizon, m = y.shape
    p = B.shape[1]

    baseline, measurement_matrix = _build_measurement_system(A, B, C, x_initial, horizon)
    measurement_rhs = y.reshape(-1) - baseline

    lhs = np.eye(horizon * p, dtype=np.float64) + tau * (
        measurement_matrix.T @ measurement_matrix
    )
    rhs = tau * (measurement_matrix.T @ measurement_rhs)

    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Reference solver mirroring AlgoTune's Kalman-filter task semantics."""
    A, B, C, y, x_initial, tau = _parse_problem(problem)
    horizon, m = y.shape
    n = A.shape[0]
    p = B.shape[1]

    w_vec = _solve_process_noise(A, B, C, y, x_initial, tau)
    w_hat = w_vec.reshape(horizon, p)

    x_hat = np.zeros((horizon + 1, n), dtype=np.float64)
    x_hat[0] = x_initial
    for t in range(horizon):
        x_hat[t + 1] = A @ x_hat[t] + B @ w_hat[t]

    v_hat = np.zeros((horizon, m), dtype=np.float64)
    for t in range(horizon):
        v_hat[t] = y[t] - C @ x_hat[t]

    return {
        "x_hat": x_hat.tolist(),
        "w_hat": w_hat.tolist(),
        "v_hat": v_hat.tolist(),
    }


def objective_value(problem: dict[str, Any], solution: dict[str, Any]) -> float:
    """Return the Kalman-filter objective value for one feasible solution."""
    _, _, _, _, _, tau = _parse_problem(problem)
    w_hat = np.asarray(solution["w_hat"], dtype=np.float64)
    v_hat = np.asarray(solution["v_hat"], dtype=np.float64)
    return float(np.sum(w_hat**2) + tau * np.sum(v_hat**2))


def relative_objective_error(reference_value: float, candidate_value: float) -> float:
    """Return the scale-aware objective mismatch."""
    scale = max(1.0, abs(reference_value))
    return float(abs(candidate_value - reference_value) / scale)


def validate_solution(problem: dict[str, Any], solution: Any) -> dict[str, float]:
    """Validate one candidate solution and return diagnostic residuals."""
    if not isinstance(solution, dict):
        raise TypeError(f"Expected dict solution, got {type(solution).__name__}.")
    required = {"x_hat", "w_hat", "v_hat"}
    if not required.issubset(solution):
        raise ValueError("Solution must contain 'x_hat', 'w_hat', and 'v_hat'.")

    A, B, C, y, x_initial, _ = _parse_problem(problem)
    horizon, m = y.shape
    n = A.shape[0]
    p = B.shape[1]

    try:
        x_hat = np.asarray(solution["x_hat"], dtype=np.float64)
        w_hat = np.asarray(solution["w_hat"], dtype=np.float64)
        v_hat = np.asarray(solution["v_hat"], dtype=np.float64)
    except Exception as exc:
        raise TypeError(f"Solution contains non-numeric data: {exc}") from exc

    if x_hat.shape != (horizon + 1, n):
        raise ValueError(f"x_hat has wrong shape: expected {(horizon + 1, n)}, got {x_hat.shape}.")
    if w_hat.shape != (horizon, p):
        raise ValueError(f"w_hat has wrong shape: expected {(horizon, p)}, got {w_hat.shape}.")
    if v_hat.shape != (horizon, m):
        raise ValueError(f"v_hat has wrong shape: expected {(horizon, m)}, got {v_hat.shape}.")
    if not (
        np.all(np.isfinite(x_hat))
        and np.all(np.isfinite(w_hat))
        and np.all(np.isfinite(v_hat))
    ):
        raise ValueError("Solution contains non-finite values.")

    initial_residual = float(np.linalg.norm(x_hat[0] - x_initial))
    if initial_residual > FEASIBILITY_TOLERANCE:
        raise ValueError("x_hat[0] does not match x_initial.")

    max_dynamics_residual = 0.0
    for t in range(horizon):
        residual = float(np.linalg.norm(x_hat[t + 1] - (A @ x_hat[t] + B @ w_hat[t])))
        max_dynamics_residual = max(max_dynamics_residual, residual)
    if max_dynamics_residual > FEASIBILITY_TOLERANCE:
        raise ValueError(
            "Dynamics constraint violated: "
            f"max residual={max_dynamics_residual}."
        )

    max_measurement_residual = 0.0
    for t in range(horizon):
        residual = float(np.linalg.norm(y[t] - (C @ x_hat[t] + v_hat[t])))
        max_measurement_residual = max(max_measurement_residual, residual)
    if max_measurement_residual > FEASIBILITY_TOLERANCE:
        raise ValueError(
            "Measurement constraint violated: "
            f"max residual={max_measurement_residual}."
        )

    candidate_objective = objective_value(problem, solution)
    reference_solution = solve_problem(problem)
    optimal_objective = objective_value(problem, reference_solution)
    if candidate_objective > optimal_objective * (1.0 + OBJECTIVE_RTOL):
        raise ValueError(
            "Kalman-filter objective mismatch: "
            f"candidate={candidate_objective}, optimal={optimal_objective}."
        )

    return {
        "candidate_objective": candidate_objective,
        "optimal_objective": optimal_objective,
        "relative_objective_error": relative_objective_error(
            optimal_objective,
            candidate_objective,
        ),
        "max_dynamics_residual": max_dynamics_residual,
        "max_measurement_residual": max_measurement_residual,
    }


def is_solution(problem: dict[str, Any], solution: Any) -> bool:
    """Return whether ``solution`` satisfies the adapted validator."""
    try:
        validate_solution(problem, solution)
    except Exception:
        return False
    return True
