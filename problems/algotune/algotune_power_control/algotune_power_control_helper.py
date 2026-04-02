"""Helper functions for the adapted AlgoTune ``power_control`` task.

This mirrors the core problem generation and validation logic from
``AlgoTune-main/AlgoTuneTasks/power_control/power_control.py`` without
importing the full AlgoTune runtime or relying on ``cvxpy``, which is not
installed in this environment.
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
    """Deterministic evaluation suite spanning small and medium networks."""
    return [
        {"n": 2, "random_seed": 1234},
        {"n": 3, "random_seed": 2024},
        {"n": 5, "random_seed": 7},
        {"n": 8, "random_seed": 31415},
        {"n": 12, "random_seed": 2718},
    ]


def generate_problem(n: int, random_seed: int) -> dict[str, Any]:
    """Generate one wireless power-control instance using AlgoTune's logic."""
    rng = np.random.default_rng(random_seed)

    G = rng.uniform(0.0, 0.2, (n, n))
    G += np.diag(rng.uniform(0.8, 1.2, n))
    sigma = rng.uniform(0.1, 0.5, n)
    P_min = rng.uniform(0.05, 0.15, n)
    P_max = rng.uniform(3.0, 7.0, n)

    P_feas = rng.uniform(P_min, P_max)
    sinr = np.array(
        [
            G[i, i] * P_feas[i] / (sigma[i] + (G[i] @ P_feas - G[i, i] * P_feas[i]))
            for i in range(n)
        ],
        dtype=np.float64,
    )
    S_min = float(0.8 * np.min(sinr))

    return {
        "G": G.tolist(),
        "sigma": sigma.tolist(),
        "P_min": P_min.tolist(),
        "P_max": P_max.tolist(),
        "S_min": S_min,
    }


def _parse_problem(
    problem: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    required = {"G", "sigma", "P_min", "P_max", "S_min"}
    missing = required.difference(problem)
    if missing:
        raise ValueError(
            f"Problem dictionary is missing keys: {', '.join(sorted(missing))}."
        )

    G = np.asarray(problem["G"], dtype=np.float64)
    sigma = np.asarray(problem["sigma"], dtype=np.float64)
    P_min = np.asarray(problem["P_min"], dtype=np.float64)
    P_max = np.asarray(problem["P_max"], dtype=np.float64)
    S_min = float(problem["S_min"])

    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square, got shape {G.shape}.")
    n = G.shape[0]
    if sigma.shape != (n,) or P_min.shape != (n,) or P_max.shape != (n,):
        raise ValueError("sigma, P_min, and P_max must all have length n.")
    if not all(
        np.all(np.isfinite(arr)) for arr in (G, sigma, P_min, P_max)
    ) or not np.isfinite(S_min):
        raise ValueError("Problem data must contain only finite values.")
    if np.any(np.diag(G) <= 0.0):
        raise ValueError("Direct channel gains must be strictly positive.")
    if np.any(P_min < 0.0) or np.any(P_max < P_min):
        raise ValueError("Power bounds are invalid.")
    if S_min <= 0.0:
        raise ValueError("S_min must be positive.")

    return G, sigma, P_min, P_max, S_min


def _sinr_values(G: np.ndarray, sigma: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.array(
        [
            G[i, i] * P[i] / (sigma[i] + (G[i] @ P - G[i, i] * P[i]))
            for i in range(P.shape[0])
        ],
        dtype=np.float64,
    )


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Solve the power-allocation LP equivalent of the SINR constraints."""
    G, sigma, P_min, P_max, S_min = _parse_problem(problem)
    n = G.shape[0]

    A_ub = np.zeros((n, n), dtype=np.float64)
    b_ub = -S_min * sigma
    for i in range(n):
        row = S_min * G[i].copy()
        row[i] = -G[i, i]
        A_ub[i] = row

    result = linprog(
        c=np.ones(n, dtype=np.float64),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=list(zip(P_min.tolist(), P_max.tolist())),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Failed to solve the power-control LP: {result.message}")

    P = np.asarray(result.x, dtype=np.float64)
    objective_value = float(np.sum(P))
    return {"P": P.tolist(), "objective": objective_value}


def validate_solution(problem: dict[str, Any], solution: Any) -> dict[str, float]:
    """Validate one candidate power allocation and return diagnostic margins."""
    if not isinstance(solution, dict):
        raise TypeError(f"Expected dict solution, got {type(solution).__name__}.")
    if "P" not in solution or "objective" not in solution:
        raise ValueError("Solution must contain 'P' and 'objective'.")

    G, sigma, P_min, P_max, S_min = _parse_problem(problem)
    P = np.asarray(solution["P"], dtype=np.float64)
    if P.shape != P_min.shape:
        raise ValueError(
            f"Candidate allocation has shape {P.shape}, expected {P_min.shape}."
        )
    if not np.all(np.isfinite(P)):
        raise ValueError("Candidate allocation must contain only finite values.")

    reported_objective = float(solution["objective"])
    candidate_objective = float(np.sum(P))
    if abs(candidate_objective - reported_objective) > OBJECTIVE_ATOL * (
        1.0 + abs(candidate_objective)
    ):
        raise ValueError("Reported objective does not match the candidate allocation.")

    min_lower_margin = float(np.min(P - P_min))
    min_upper_margin = float(np.min(P_max - P))
    if min_lower_margin < -FEASIBILITY_ATOL or min_upper_margin < -FEASIBILITY_ATOL:
        raise ValueError("Power allocation violates box constraints.")

    sinr = _sinr_values(G, sigma, P)
    min_sinr_margin = float(np.min(sinr - S_min))
    if min_sinr_margin < -FEASIBILITY_ATOL:
        raise ValueError(f"SINR constraints violated by {-min_sinr_margin:.3e}.")

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
        "min_sinr_margin": min_sinr_margin,
        "min_bound_margin": float(min(min_lower_margin, min_upper_margin)),
    }


def is_solution(problem: dict[str, Any], solution: Any) -> bool:
    """Return whether ``solution`` satisfies the adapted validator."""
    try:
        validate_solution(problem, solution)
    except Exception:
        return False
    return True
