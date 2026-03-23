"""Helper functions for the adapted AlgoTune ``ode_lotkavolterra`` task."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from scipy.integrate import solve_ivp

SOLUTION_RTOL = 1.0e-5
SOLUTION_ATOL = 1.0e-8


class CaseSpec(TypedDict):
    n: int
    random_seed: int


def get_case_specs() -> list[CaseSpec]:
    """Deterministic evaluation suite spanning several integration horizons."""
    return [
        {"n": 1, "random_seed": 1234},
        {"n": 2, "random_seed": 2024},
        {"n": 4, "random_seed": 7},
        {"n": 6, "random_seed": 31415},
        {"n": 8, "random_seed": 2718},
    ]


def generate_problem(n: int = 1, random_seed: int = 1234) -> dict[str, Any]:
    """Generate one Lotka-Volterra instance using AlgoTune's logic."""
    np.random.seed(random_seed)

    alpha = 1.1 * np.random.uniform(0.8, 1.2)
    beta = 0.4 * np.random.uniform(0.8, 1.2)
    delta = 0.1 * np.random.uniform(0.8, 1.2)
    gamma = 0.4 * np.random.uniform(0.8, 1.2)

    prey_init = 10.0 * np.random.uniform(0.8, 1.2)
    predator_init = 5.0 * np.random.uniform(0.8, 1.2)

    return {
        "t0": 0.0,
        "t1": float(n),
        "y0": [prey_init, predator_init],
        "params": {
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "gamma": gamma,
        },
    }


def _parse_problem(problem: dict[str, Any]) -> tuple[float, float, np.ndarray, dict[str, float]]:
    required = {"t0", "t1", "y0", "params"}
    missing = required.difference(problem)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"Problem dictionary is missing keys: {missing_keys}.")

    t0 = float(problem["t0"])
    t1 = float(problem["t1"])
    y0 = np.asarray(problem["y0"], dtype=np.float64)
    params = dict(problem["params"])

    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        raise ValueError("Problem must define finite times with t1 > t0.")
    if y0.shape != (2,):
        raise ValueError(f"y0 must have shape (2,), got {y0.shape}.")
    if not np.all(np.isfinite(y0)):
        raise ValueError("y0 must contain only finite values.")
    if np.any(y0 < 0.0):
        raise ValueError("y0 must contain non-negative populations.")

    param_keys = {"alpha", "beta", "delta", "gamma"}
    if set(params) != param_keys:
        raise ValueError(f"params must contain exactly {sorted(param_keys)}.")
    parsed_params = {key: float(params[key]) for key in param_keys}
    if not all(np.isfinite(value) for value in parsed_params.values()):
        raise ValueError("params must contain only finite scalars.")

    return t0, t1, y0, parsed_params


def solve_problem(problem: dict[str, Any]) -> list[float]:
    """Reference solver mirroring AlgoTune's Lotka-Volterra task semantics."""
    t0, t1, y0, params = _parse_problem(problem)

    def lotka_volterra(_t: float, y: np.ndarray) -> np.ndarray:
        prey, predator = y
        return np.array(
            [
                params["alpha"] * prey - params["beta"] * prey * predator,
                params["delta"] * prey * predator - params["gamma"] * predator,
            ],
            dtype=np.float64,
        )

    sol = solve_ivp(
        lotka_volterra,
        [t0, t1],
        y0,
        method="RK45",
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")
    return sol.y[:, -1].tolist()


def relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return the scale-aware relative L2 error."""
    return float(
        np.linalg.norm(candidate - reference) / (np.linalg.norm(reference) + 1.0e-12)
    )


def validate_solution(problem: dict[str, Any], solution: Any) -> dict[str, float]:
    """Validate one candidate final state and return diagnostics."""
    candidate = np.asarray(solution, dtype=np.float64)
    _, _, y0, _ = _parse_problem(problem)

    if candidate.shape != y0.shape:
        raise ValueError(f"Output has wrong shape: expected {y0.shape}, got {candidate.shape}.")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("Output must contain only finite values.")
    if np.any(candidate < 0.0):
        raise ValueError("Output must contain non-negative populations.")

    reference = np.asarray(solve_problem(problem), dtype=np.float64)
    err = relative_error(reference, candidate)
    if not np.allclose(candidate, reference, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL):
        raise ValueError(f"Final state mismatch: relative_error={err}.")
    return {"relative_error": err}


def is_solution(problem: dict[str, Any], solution: Any) -> bool:
    """Return whether ``solution`` satisfies the adapted validator."""
    try:
        validate_solution(problem, solution)
    except Exception:
        return False
    return True
