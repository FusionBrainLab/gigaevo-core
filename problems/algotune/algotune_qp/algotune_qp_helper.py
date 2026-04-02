"""Helper functions for the adapted AlgoTune ``qp`` task.

This mirrors the core problem generation and validation logic from
``AlgoTune-main/AlgoTuneTasks/qp/qp.py`` without importing the full AlgoTune
runtime or relying on ``cvxpy``, which is not installed in this environment.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

FEASIBILITY_ATOL = 1.0e-6
OBJECTIVE_ATOL = 1.0e-6
OBJECTIVE_RTOL = 1.0e-5


class CaseSpec(TypedDict):
    n: int
    random_seed: int


def get_case_specs() -> list[CaseSpec]:
    """Deterministic evaluation suite spanning small and medium QPs."""
    return [
        {"n": 2, "random_seed": 1234},
        {"n": 3, "random_seed": 2024},
        {"n": 4, "random_seed": 7},
        {"n": 6, "random_seed": 31415},
        {"n": 8, "random_seed": 2718},
    ]


def generate_problem(n: int, random_seed: int = 1) -> dict[str, Any]:
    """Generate one feasible convex quadratic program using AlgoTune's logic."""
    n = max(int(n), 2)
    rng = np.random.default_rng(random_seed)
    m = p = n // 2

    M = rng.standard_normal((n, n))
    P = M.T @ M
    q = rng.standard_normal(n)
    G = rng.standard_normal((m, n))
    A = rng.standard_normal((p, n))

    x_feas = rng.standard_normal(n)
    h = G @ x_feas + np.abs(rng.standard_normal(m))
    b = A @ x_feas

    return {
        "P": P.tolist(),
        "q": q.tolist(),
        "G": G.tolist(),
        "h": h.tolist(),
        "A": A.tolist(),
        "b": b.tolist(),
    }


def _parse_problem(
    problem: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"P", "q", "G", "h", "A", "b"}
    missing = required.difference(problem)
    if missing:
        raise ValueError(
            f"Problem dictionary is missing keys: {', '.join(sorted(missing))}."
        )

    P = np.asarray(problem["P"], dtype=np.float64)
    q = np.asarray(problem["q"], dtype=np.float64)
    G = np.asarray(problem["G"], dtype=np.float64)
    h = np.asarray(problem["h"], dtype=np.float64)
    A = np.asarray(problem["A"], dtype=np.float64)
    b = np.asarray(problem["b"], dtype=np.float64)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P must be square, got shape {P.shape}.")
    n = P.shape[0]
    if q.shape != (n,):
        raise ValueError(f"q must have shape {(n,)}, got {q.shape}.")
    if G.ndim != 2 or G.shape[1] != n:
        raise ValueError(f"G must have shape (m, {n}), got {G.shape}.")
    if h.shape != (G.shape[0],):
        raise ValueError(f"h must have shape {(G.shape[0],)}, got {h.shape}.")
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError(f"A must have shape (p, {n}), got {A.shape}.")
    if b.shape != (A.shape[0],):
        raise ValueError(f"b must have shape {(A.shape[0],)}, got {b.shape}.")

    arrays = (P, q, G, h, A, b)
    if not all(np.all(np.isfinite(arr)) for arr in arrays):
        raise ValueError("All problem arrays must contain only finite values.")

    P = 0.5 * (P + P.T)
    return P, q, G, h, A, b


def _objective_value(P: np.ndarray, q: np.ndarray, x: np.ndarray) -> float:
    return float(0.5 * x @ P @ x + q @ x)


def _find_feasible_point(
    G: np.ndarray, h: np.ndarray, A: np.ndarray, b: np.ndarray
) -> np.ndarray:
    n = G.shape[1]
    result = linprog(
        c=np.zeros(n, dtype=np.float64),
        A_ub=G,
        b_ub=h,
        A_eq=A,
        b_eq=b,
        bounds=[(None, None)] * n,
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Failed to recover a feasible point for the QP: {result.message}"
        )
    return np.asarray(result.x, dtype=np.float64)


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Solve the convex QP with SciPy's constrained optimizer."""
    P, q, G, h, A, b = _parse_problem(problem)
    n = P.shape[0]
    x0 = _find_feasible_point(G, h, A, b)

    constraints = [
        LinearConstraint(G, -np.inf, h),
        LinearConstraint(A, b, b),
    ]
    bounds = Bounds(lb=np.full(n, -np.inf), ub=np.full(n, np.inf))

    def objective(x: np.ndarray) -> float:
        return _objective_value(P, q, x)

    def gradient(x: np.ndarray) -> np.ndarray:
        return P @ x + q

    def hessian(_: np.ndarray) -> np.ndarray:
        return P

    result = minimize(
        objective,
        x0=x0,
        method="trust-constr",
        jac=gradient,
        hess=hessian,
        constraints=constraints,
        bounds=bounds,
        options={
            "gtol": 1.0e-10,
            "xtol": 1.0e-10,
            "barrier_tol": 1.0e-12,
            "maxiter": 1000,
        },
    )

    if (not result.success or result.x is None) and x0 is not None:
        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            jac=gradient,
            constraints=constraints,
            bounds=bounds,
            options={"ftol": 1.0e-12, "maxiter": 1000},
        )

    if not result.success or result.x is None:
        raise RuntimeError(f"Failed to solve the QP: {result.message}")

    x = np.asarray(result.x, dtype=np.float64)
    if np.max(G @ x - h, initial=-np.inf) > 1.0e-5:
        raise RuntimeError("QP solver returned an infeasible inequality solution.")
    if not np.allclose(A @ x, b, atol=1.0e-6, rtol=0.0):
        raise RuntimeError("QP solver returned an infeasible equality solution.")

    objective_value = _objective_value(P, q, x)
    return {"solution": x.tolist(), "objective": objective_value}


def validate_solution(problem: dict[str, Any], solution: Any) -> dict[str, float]:
    """Validate one candidate QP solution and return objective diagnostics."""
    if not isinstance(solution, dict):
        raise TypeError(f"Expected dict solution, got {type(solution).__name__}.")
    if "solution" not in solution or "objective" not in solution:
        raise ValueError("Solution must contain 'solution' and 'objective'.")

    P, q, G, h, A, b = _parse_problem(problem)
    x = np.asarray(solution["solution"], dtype=np.float64)
    if x.shape != (P.shape[0],):
        raise ValueError(
            f"Candidate solution has shape {x.shape}, expected {(P.shape[0],)}."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("Candidate solution must contain only finite values.")

    reported_objective = float(solution["objective"])
    candidate_objective = _objective_value(P, q, x)
    if abs(candidate_objective - reported_objective) > OBJECTIVE_ATOL * (
        1.0 + abs(candidate_objective)
    ):
        raise ValueError("Reported objective does not match the candidate point.")

    max_inequality_violation = float(np.max(G @ x - h, initial=-np.inf))
    if max_inequality_violation > FEASIBILITY_ATOL:
        raise ValueError(
            f"Inequality constraints violated by {max_inequality_violation:.3e}."
        )

    equality_residual = float(np.max(np.abs(A @ x - b), initial=0.0))
    if equality_residual > FEASIBILITY_ATOL:
        raise ValueError(f"Equality constraints violated by {equality_residual:.3e}.")

    reference = solve_problem(problem)
    reference_objective = float(reference["objective"])
    objective_gap = candidate_objective - reference_objective
    if objective_gap > OBJECTIVE_ATOL + OBJECTIVE_RTOL * (
        1.0 + abs(reference_objective)
    ):
        raise ValueError(
            f"Objective is suboptimal by {objective_gap:.3e} relative to the reference optimum."
        )

    return {
        "objective_gap": float(objective_gap),
        "max_inequality_violation": max_inequality_violation,
        "max_equality_residual": equality_residual,
    }


def is_solution(problem: dict[str, Any], solution: Any) -> bool:
    """Return whether ``solution`` satisfies the adapted validator."""
    try:
        validate_solution(problem, solution)
    except Exception:
        return False
    return True
