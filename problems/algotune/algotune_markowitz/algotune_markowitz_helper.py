"""Helper functions for the adapted AlgoTune ``markowitz`` task.

This mirrors the core problem generation and validation logic from
``AlgoTune-main/AlgoTuneTasks/markowitz/markowitz.py`` without importing the
full AlgoTune runtime or relying on ``cvxpy``, which is not installed in this
environment.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

FEASIBILITY_ATOL = 1.0e-6
OBJECTIVE_ATOL = 1.0e-6
OBJECTIVE_RTOL = 1.0e-5


class CaseSpec(TypedDict):
    n: int
    random_seed: int


def get_case_specs() -> list[CaseSpec]:
    """Deterministic evaluation suite spanning small and medium portfolios."""
    return [
        {"n": 2, "random_seed": 1234},
        {"n": 4, "random_seed": 2024},
        {"n": 6, "random_seed": 7},
        {"n": 8, "random_seed": 31415},
        {"n": 12, "random_seed": 2718},
    ]


def generate_problem(n: int, random_seed: int) -> dict[str, Any]:
    """Generate one long-only Markowitz instance using AlgoTune's logic."""
    rng = np.random.default_rng(random_seed)

    mu = rng.standard_normal(n) * 0.1 + 0.05
    Z = rng.standard_normal((n, n)) / np.sqrt(n)
    sigma = Z.T @ Z + 1.0e-4 * np.eye(n)
    gamma = 1.0

    return {"mu": mu.tolist(), "sigma": sigma.tolist(), "gamma": gamma}


def _parse_problem(problem: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    required = {"mu", "sigma", "gamma"}
    missing = required.difference(problem)
    if missing:
        raise ValueError(
            f"Problem dictionary is missing keys: {', '.join(sorted(missing))}."
        )

    mu = np.asarray(problem["mu"], dtype=np.float64)
    sigma = np.asarray(problem["sigma"], dtype=np.float64)
    gamma = float(problem["gamma"])

    if mu.ndim != 1:
        raise ValueError(f"mu must be one-dimensional, got shape {mu.shape}.")
    n = mu.shape[0]
    if sigma.shape != (n, n):
        raise ValueError(f"sigma must have shape {(n, n)}, got {sigma.shape}.")
    if (
        not np.all(np.isfinite(mu))
        or not np.all(np.isfinite(sigma))
        or not np.isfinite(gamma)
    ):
        raise ValueError("Problem data must contain only finite values.")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")

    sigma = 0.5 * (sigma + sigma.T)
    return mu, sigma, gamma


def _objective_value(
    mu: np.ndarray, sigma: np.ndarray, gamma: float, w: np.ndarray
) -> float:
    return float(mu @ w - gamma * (w @ sigma @ w))


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Solve the long-only Markowitz problem with SciPy."""
    mu, sigma, gamma = _parse_problem(problem)
    n = mu.shape[0]
    x0 = np.full(n, 1.0 / n, dtype=np.float64)

    equality_constraint = LinearConstraint(
        np.ones((1, n), dtype=np.float64), [1.0], [1.0]
    )
    bounds = Bounds(lb=np.zeros(n, dtype=np.float64), ub=np.ones(n, dtype=np.float64))

    def objective(w: np.ndarray) -> float:
        return float(gamma * (w @ sigma @ w) - mu @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * gamma * (sigma @ w) - mu

    def hessian(_: np.ndarray) -> np.ndarray:
        return 2.0 * gamma * sigma

    result = minimize(
        objective,
        x0=x0,
        method="trust-constr",
        jac=gradient,
        hess=hessian,
        constraints=[equality_constraint],
        bounds=bounds,
        options={
            "gtol": 1.0e-10,
            "xtol": 1.0e-10,
            "barrier_tol": 1.0e-12,
            "maxiter": 1000,
        },
    )

    if not result.success or result.x is None:
        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            jac=gradient,
            constraints=[equality_constraint],
            bounds=bounds,
            options={"ftol": 1.0e-12, "maxiter": 1000},
        )

    if not result.success or result.x is None:
        raise RuntimeError(f"Failed to solve the Markowitz problem: {result.message}")

    w = np.asarray(result.x, dtype=np.float64)
    if abs(float(np.sum(w)) - 1.0) > 1.0e-6:
        raise RuntimeError(
            "Solver returned a portfolio whose weights do not sum to one."
        )
    if np.min(w) < -1.0e-6:
        raise RuntimeError("Solver returned a portfolio with negative weights.")

    objective_value = _objective_value(mu, sigma, gamma, w)
    return {"w": w.tolist(), "objective": objective_value}


def validate_solution(problem: dict[str, Any], solution: Any) -> dict[str, float]:
    """Validate one candidate portfolio and return objective diagnostics."""
    if not isinstance(solution, dict):
        raise TypeError(f"Expected dict solution, got {type(solution).__name__}.")
    if "w" not in solution or "objective" not in solution:
        raise ValueError("Solution must contain 'w' and 'objective'.")

    mu, sigma, gamma = _parse_problem(problem)
    w = np.asarray(solution["w"], dtype=np.float64)
    if w.shape != (mu.shape[0],):
        raise ValueError(
            f"Candidate portfolio has shape {w.shape}, expected {(mu.shape[0],)}."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("Candidate portfolio must contain only finite values.")

    reported_objective = float(solution["objective"])
    candidate_objective = _objective_value(mu, sigma, gamma, w)
    if abs(candidate_objective - reported_objective) > OBJECTIVE_ATOL * (
        1.0 + abs(candidate_objective)
    ):
        raise ValueError("Reported objective does not match the candidate portfolio.")

    simplex_residual = abs(float(np.sum(w)) - 1.0)
    if simplex_residual > FEASIBILITY_ATOL:
        raise ValueError(
            f"Portfolio weights do not sum to one: residual {simplex_residual:.3e}."
        )

    min_weight = float(np.min(w))
    if min_weight < -FEASIBILITY_ATOL:
        raise ValueError(f"Portfolio has negative weight {min_weight:.3e}.")

    reference = solve_problem(problem)
    reference_objective = float(reference["objective"])
    objective_gap = reference_objective - candidate_objective
    if objective_gap > OBJECTIVE_ATOL + OBJECTIVE_RTOL * (
        1.0 + abs(reference_objective)
    ):
        raise ValueError(
            f"Objective is suboptimal by {objective_gap:.3e} relative to the reference optimum."
        )

    return {
        "objective_gap": float(objective_gap),
        "simplex_residual": float(simplex_residual),
        "min_weight": min_weight,
    }


def is_solution(problem: dict[str, Any], solution: Any) -> bool:
    """Return whether ``solution`` satisfies the adapted validator."""
    try:
        validate_solution(problem, solution)
    except Exception:
        return False
    return True
