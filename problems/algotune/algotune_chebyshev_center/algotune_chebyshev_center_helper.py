"""Helper functions for the adapted AlgoTune ``chebyshev_center`` task.

This mirrors the core problem generation and validation logic from
``AlgoTune-main/AlgoTuneTasks/chebyshev_center/chebyshev_center.py`` without
importing the full AlgoTune runtime or relying on ``cvxpy``, which is not
installed in this environment.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from scipy.optimize import linprog

FEASIBILITY_ATOL = 1.0e-6
RADIUS_ATOL = 1.0e-6
RADIUS_RTOL = 1.0e-6


class CaseSpec(TypedDict):
    n: int
    random_seed: int


def get_case_specs() -> list[CaseSpec]:
    """Deterministic evaluation suite spanning small and medium polyhedra."""
    return [
        {"n": 1, "random_seed": 1234},
        {"n": 2, "random_seed": 2024},
        {"n": 4, "random_seed": 7},
        {"n": 6, "random_seed": 31415},
        {"n": 8, "random_seed": 2718},
    ]


def generate_problem(n: int, random_seed: int = 1) -> dict[str, Any]:
    """Generate one Chebyshev-center instance using AlgoTune's logic."""
    rng = np.random.RandomState(random_seed)

    n = int(n) + 1
    m = n // 2
    x_pseudo = rng.randn(n)
    a = rng.randn(m, n)
    a = np.concatenate([a, -a], axis=0)
    b = a @ x_pseudo + rng.rand(a.shape[0])

    return {"a": a.tolist(), "b": b.tolist()}


def _parse_problem(problem: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    required = {"a", "b"}
    missing = required.difference(problem)
    if missing:
        raise ValueError(f"Problem dictionary is missing keys: {', '.join(sorted(missing))}.")

    a = np.asarray(problem["a"], dtype=np.float64)
    b = np.asarray(problem["b"], dtype=np.float64)

    if a.ndim != 2:
        raise ValueError(f"a must be a matrix, got shape {a.shape}.")
    if b.shape != (a.shape[0],):
        raise ValueError(f"b must have shape {(a.shape[0],)}, got {b.shape}.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Problem data must contain only finite values.")

    row_norms = np.linalg.norm(a, axis=1)
    if np.any(row_norms <= 0.0):
        raise ValueError("Each row of a must have non-zero norm.")
    return a, b


def _solve_reference(problem: dict[str, Any]) -> tuple[np.ndarray, float]:
    a, b = _parse_problem(problem)
    n = a.shape[1]
    row_norms = np.linalg.norm(a, axis=1)

    c = np.zeros(n + 1, dtype=np.float64)
    c[-1] = -1.0
    A_ub = np.hstack((a, row_norms[:, None]))
    bounds = [(None, None)] * n + [(0.0, None)]

    result = linprog(c=c, A_ub=A_ub, b_ub=b, bounds=bounds, method="highs")
    if not result.success or result.x is None:
        raise RuntimeError(f"Failed to solve the Chebyshev-center LP: {result.message}")

    x = np.asarray(result.x[:n], dtype=np.float64)
    radius = float(result.x[-1])
    return x, radius


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Solve the Chebyshev-center problem and return the center point."""
    x, _ = _solve_reference(problem)
    return {"solution": x.tolist()}


def validate_solution(problem: dict[str, Any], solution: Any) -> dict[str, float]:
    """Validate one candidate center and return radius diagnostics."""
    if not isinstance(solution, dict):
        raise TypeError(f"Expected dict solution, got {type(solution).__name__}.")
    if "solution" not in solution:
        raise ValueError("Solution must contain 'solution'.")

    a, b = _parse_problem(problem)
    x = np.asarray(solution["solution"], dtype=np.float64)
    if x.shape != (a.shape[1],):
        raise ValueError(f"Candidate center has shape {x.shape}, expected {(a.shape[1],)}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Candidate center must contain only finite values.")

    row_norms = np.linalg.norm(a, axis=1)
    max_linear_violation = float(np.max(a @ x - b, initial=-np.inf))
    if max_linear_violation > FEASIBILITY_ATOL:
        raise ValueError(f"Polyhedron constraints violated by {max_linear_violation:.3e}.")

    candidate_radius = float(np.min((b - a @ x) / row_norms, initial=np.inf))
    if candidate_radius < -FEASIBILITY_ATOL:
        raise ValueError("Candidate point lies outside the feasible polyhedron.")

    _, reference_radius = _solve_reference(problem)
    radius_gap = reference_radius - candidate_radius
    if radius_gap > RADIUS_ATOL + RADIUS_RTOL * (1.0 + abs(reference_radius)):
        raise ValueError(
            f"Inscribed-ball radius is suboptimal by {radius_gap:.3e} relative to the optimum."
        )

    return {
        "radius_gap": float(radius_gap),
        "candidate_radius": candidate_radius,
        "max_linear_violation": max_linear_violation,
    }


def is_solution(problem: dict[str, Any], solution: Any) -> bool:
    """Return whether ``solution`` satisfies the adapted validator."""
    try:
        validate_solution(problem, solution)
    except Exception:
        return False
    return True
