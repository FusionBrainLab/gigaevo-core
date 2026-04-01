"""Helper functions for the adapted AlgoTune ``pde_heat1d`` task."""

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
    """Deterministic evaluation suite spanning several spatial resolutions."""
    return [
        {"n": 1, "random_seed": 1234},
        {"n": 2, "random_seed": 2024},
        {"n": 4, "random_seed": 7},
        {"n": 6, "random_seed": 31415},
        {"n": 8, "random_seed": 2718},
    ]


def generate_problem(n: int = 1, random_seed: int = 1234) -> dict[str, Any]:
    """Generate one heat-equation instance using AlgoTune's logic."""
    np.random.seed(random_seed)

    x_min = 0.0
    x_max = 1.0
    alpha = 0.01 * np.random.uniform(0.8, 1.2)

    num_points = 20 * n
    dx = (x_max - x_min) / (num_points + 1)
    x_grid = np.linspace(x_min + dx, x_max - dx, num_points)

    u_init = np.zeros(num_points, dtype=np.float64)
    num_bumps = np.random.randint(2, 5)
    for _ in range(num_bumps):
        center = np.random.uniform(0.2, 0.8)
        width = np.random.uniform(0.05, 0.15)
        sign = np.random.choice([-1, 1])
        amplitude = sign * np.random.uniform(0.8, 1.0)
        distance = np.abs(x_grid - center)
        u_init += amplitude * np.exp(-((distance / width) ** 2))

    if np.max(np.abs(u_init)) > 0.0:
        u_init = u_init / np.max(np.abs(u_init))

    return {
        "t0": 0.0,
        "t1": 2.0,
        "y0": u_init.tolist(),
        "params": {"alpha": alpha, "dx": dx, "num_points": int(num_points)},
        "x_grid": x_grid.tolist(),
    }


def _parse_problem(
    problem: dict[str, Any],
) -> tuple[float, float, np.ndarray, dict[str, float]]:
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
    if y0.ndim != 1 or y0.size == 0:
        raise ValueError("y0 must be a non-empty 1D array.")
    if not np.all(np.isfinite(y0)):
        raise ValueError("y0 must contain only finite values.")

    param_keys = {"alpha", "dx", "num_points"}
    if set(params) != param_keys:
        raise ValueError(f"params must contain exactly {sorted(param_keys)}.")

    alpha = float(params["alpha"])
    dx = float(params["dx"])
    num_points = int(params["num_points"])
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be a finite positive scalar.")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be a finite positive scalar.")
    if num_points != y0.size:
        raise ValueError("num_points must match the length of y0.")

    return t0, t1, y0, {"alpha": alpha, "dx": dx, "num_points": float(num_points)}


def solve_problem(problem: dict[str, Any]) -> list[float]:
    """Reference solver mirroring AlgoTune's heat-equation task semantics."""
    t0, t1, y0, params = _parse_problem(problem)
    alpha = params["alpha"]
    dx = params["dx"]

    def heat_equation(_t: float, u: np.ndarray) -> np.ndarray:
        u_padded = np.pad(u, 1, mode="constant", constant_values=0.0)
        u_xx = (u_padded[2:] - 2.0 * u_padded[1:-1] + u_padded[:-2]) / (dx**2)
        return alpha * u_xx

    sol = solve_ivp(
        heat_equation,
        [t0, t1],
        y0,
        method="RK45",
        rtol=1.0e-6,
        atol=1.0e-6,
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
        raise ValueError(
            f"Output has wrong shape: expected {y0.shape}, got {candidate.shape}."
        )
    if not np.all(np.isfinite(candidate)):
        raise ValueError("Output must contain only finite values.")

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
