"""Helper functions for the adapted AlgoTune ``lp_box`` task.

This mirrors the core problem generation and validation logic from
``AlgoTune-main/AlgoTuneTasks/lp_box/lp_box.py`` without importing the full
AlgoTune runtime or relying on ``cvxpy``, which is not installed in this
environment.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from scipy.optimize import linprog

FEASIBILITY_ATOL = 1.0e-6
OBJECTIVE_ATOL = 1.0e-6
OBJECTIVE_RTOL = 1.0e-6


class CaseSpec(TypedDict):
    n: int
    random_seed: int


def get_case_specs() -> list[CaseSpec]:
    """Deterministic evaluation suite spanning small and medium LPs."""
    return [
        {"n": 1, "random_seed": 1234},
        {"n": 2, "random_seed": 2024},
        {"n": 4, "random_seed": 7},
        {"n": 6, "random_seed": 31415},
        {"n": 8, "random_seed": 2718},
    ]


def generate_problem(n: int, random_seed: int = 1) -> dict[str, Any]:
    """Generate one boxed linear program using AlgoTune's logic."""
    rng = np.random.RandomState(random_seed)

    n = int(n) + 1
    m = n // 2
    A = rng.rand(m, n)
    b = A.dot(np.ones(n, dtype=np.float64)) / 2.0
    c = rng.randn(n)

    return {"c": c.tolist(), "A": A.tolist(), "b": b.tolist()}


def _parse_problem(problem: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"c", "A", "b"}
    missing = required.difference(problem)
    if missing:
        raise ValueError(f"Problem dictionary is missing keys: {', '.join(sorted(missing))}.")

    c = np.asarray(problem["c"], dtype=np.float64)
    A = np.asarray(problem["A"], dtype=np.float64)
    b = np.asarray(problem["b"], dtype=np.float64)

    if c.ndim != 1:
        raise ValueError(f"c must be one-dimensional, got shape {c.shape}.")
    n = c.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError(f"A must have shape (m, {n}), got {A.shape}.")
    if b.shape != (A.shape[0],):
        raise ValueError(f"b must have shape {(A.shape[0],)}, got {b.shape}.")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("Problem data must contain only finite values.")

    return c, A, b


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Solve the boxed linear program with SciPy's HiGHS backend."""
    c, A, b = _parse_problem(problem)
    result = linprog(
        c=c,
        A_ub=A,
        b_ub=b,
        bounds=[(0.0, 1.0)] * c.shape[0],
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Failed to solve the boxed LP: {result.message}")

    x = np.asarray(result.x, dtype=np.float64)
    objective_value = float(c @ x)
    return {"solution": x.tolist(), "objective": objective_value}


def validate_solution(problem: dict[str, Any], solution: Any) -> dict[str, float]:
    """Validate one candidate LP-box solution and return objective diagnostics."""
    if not isinstance(solution, dict):
        raise TypeError(f"Expected dict solution, got {type(solution).__name__}.")
    if "solution" not in solution or "objective" not in solution:
        raise ValueError("Solution must contain 'solution' and 'objective'.")

    c, A, b = _parse_problem(problem)
    x = np.asarray(solution["solution"], dtype=np.float64)
    if x.shape != (c.shape[0],):
        raise ValueError(f"Candidate solution has shape {x.shape}, expected {(c.shape[0],)}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Candidate solution must contain only finite values.")

    reported_objective = float(solution["objective"])
    candidate_objective = float(c @ x)
    if abs(candidate_objective - reported_objective) > OBJECTIVE_ATOL * (
        1.0 + abs(candidate_objective)
    ):
        raise ValueError("Reported objective does not match the candidate point.")

    max_linear_violation = float(np.max(A @ x - b, initial=-np.inf))
    if max_linear_violation > FEASIBILITY_ATOL:
        raise ValueError(f"Linear constraints violated by {max_linear_violation:.3e}.")

    lower_bound_violation = float(max(-np.min(x), 0.0))
    upper_bound_violation = float(max(np.max(x - 1.0), 0.0))
    if lower_bound_violation > FEASIBILITY_ATOL or upper_bound_violation > FEASIBILITY_ATOL:
        raise ValueError("Box constraints 0 <= x <= 1 are violated.")

    reference = solve_problem(problem)
    reference_objective = float(reference["objective"])
    objective_gap = candidate_objective - reference_objective
    if objective_gap > OBJECTIVE_ATOL + OBJECTIVE_RTOL * (1.0 + abs(reference_objective)):
        raise ValueError(
            f"Objective is suboptimal by {objective_gap:.3e} relative to the reference optimum."
        )

    return {
        "objective_gap": float(objective_gap),
        "max_linear_violation": max_linear_violation,
        "max_box_violation": float(max(lower_bound_violation, upper_bound_violation)),
    }


def is_solution(problem: dict[str, Any], solution: Any) -> bool:
    """Return whether ``solution`` satisfies the adapted validator."""
    try:
        validate_solution(problem, solution)
    except Exception:
        return False
    return True
