"""Helper functions for the adapted AlgoTune ``battery_scheduling`` task.

This mirrors the core problem generation and validation logic from
``AlgoTune-main/AlgoTuneTasks/battery_scheduling/battery_scheduling.py``
without importing the full AlgoTune runtime or relying on ``cvxpy``, which is
not installed in this environment.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from scipy.optimize import linprog

FEASIBILITY_ATOL = 1.0e-6
COST_ATOL = 1.0e-6
COST_RTOL = 1.0e-6


class CaseSpec(TypedDict):
    n: int
    random_seed: int


def get_case_specs() -> list[CaseSpec]:
    """Deterministic evaluation suite spanning short and medium horizons."""
    return [
        {"n": 1, "random_seed": 1234},
        {"n": 2, "random_seed": 2024},
        {"n": 3, "random_seed": 7},
        {"n": 4, "random_seed": 31415},
        {"n": 5, "random_seed": 2718},
    ]


def generate_problem(n: int, random_seed: int = 1) -> dict[str, Any]:
    """Generate one battery-scheduling instance using AlgoTune's logic."""
    rng = np.random.RandomState(random_seed)

    base_T = 24
    days = max(1, int(n))
    T = base_T * days

    t = np.arange(T, dtype=np.int64)
    hours_of_day = t % 24

    base_pattern = np.sin((hours_of_day - 6) * np.pi / 12.0) + 0.5
    base_pattern = np.where(base_pattern > 0.0, base_pattern, 0.0)

    daily_factors = 1.0 + 0.2 * rng.uniform(-1.0, 1.0, days)
    price_factors = np.repeat(daily_factors, base_T)
    price_noise = 0.1 * rng.normal(0.0, 1.0, T)
    p = 10.0 * (base_pattern * price_factors + price_noise)
    p = np.maximum(p, 1.0)

    base_demand = np.sin((hours_of_day - 8) * np.pi / 12.0) + 1.2
    base_demand = np.where(base_demand > 0.0, base_demand, 0.2)

    demand_factors = 1.0 + 0.15 * rng.uniform(-1.0, 1.0, days)
    demand_factors = np.repeat(demand_factors, base_T)
    demand_noise = 0.05 * rng.normal(0.0, 1.0, T)
    u = 5.0 * (base_demand * demand_factors + demand_noise)
    u = np.maximum(u, 0.1)

    Q = 25.0
    C = 4.0
    D = 4.0
    efficiency = 0.9

    return {
        "T": T,
        "p": p.tolist(),
        "u": u.tolist(),
        "batteries": [
            {
                "Q": Q,
                "C": C,
                "D": D,
                "efficiency": efficiency,
            }
        ],
        "deg_cost": 0.0,
        "num_batteries": 1,
    }


def _parse_problem(
    problem: dict[str, Any],
) -> tuple[int, np.ndarray, np.ndarray, dict[str, float], float, int]:
    required = {"T", "p", "u", "batteries", "deg_cost", "num_batteries"}
    missing = required.difference(problem)
    if missing:
        raise ValueError(
            f"Problem dictionary is missing keys: {', '.join(sorted(missing))}."
        )

    T = int(problem["T"])
    p = np.asarray(problem["p"], dtype=np.float64)
    u = np.asarray(problem["u"], dtype=np.float64)
    batteries = problem["batteries"]
    deg_cost = float(problem["deg_cost"])
    num_batteries = int(problem["num_batteries"])

    if T <= 0:
        raise ValueError("T must be positive.")
    if p.shape != (T,) or u.shape != (T,):
        raise ValueError(
            f"p and u must both have shape {(T,)}, got {p.shape} and {u.shape}."
        )
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(u)):
        raise ValueError("p and u must contain only finite values.")
    if not isinstance(batteries, list) or len(batteries) != 1 or num_batteries != 1:
        raise ValueError("This adapted task supports exactly one battery.")
    if deg_cost != 0.0:
        raise ValueError("This adapted task expects zero degradation cost.")

    battery = batteries[0]
    Q = float(battery["Q"])
    C = float(battery["C"])
    D = float(battery["D"])
    efficiency = float(battery["efficiency"])
    if min(Q, C, D, efficiency) <= 0.0:
        raise ValueError("Battery parameters must be strictly positive.")

    return (
        T,
        p,
        u,
        {"Q": Q, "C": C, "D": D, "efficiency": efficiency},
        deg_cost,
        num_batteries,
    )


def _solve_lp(problem: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, p, u, battery, _, _ = _parse_problem(problem)
    Q = battery["Q"]
    C = battery["C"]
    D = battery["D"]
    efficiency = battery["efficiency"]

    num_vars = 3 * T
    q_slice = slice(0, T)
    c_in_slice = slice(T, 2 * T)
    c_out_slice = slice(2 * T, 3 * T)

    c = np.zeros(num_vars, dtype=np.float64)
    c[c_in_slice] = p
    c[c_out_slice] = -p

    A_eq = np.zeros((T, num_vars), dtype=np.float64)
    b_eq = np.zeros(T, dtype=np.float64)
    for t in range(T - 1):
        A_eq[t, q_slice.start + t + 1] = 1.0
        A_eq[t, q_slice.start + t] = -1.0
        A_eq[t, c_in_slice.start + t] = -efficiency
        A_eq[t, c_out_slice.start + t] = 1.0 / efficiency
    A_eq[T - 1, q_slice.start + 0] = 1.0
    A_eq[T - 1, q_slice.start + T - 1] = -1.0
    A_eq[T - 1, c_in_slice.start + T - 1] = -efficiency
    A_eq[T - 1, c_out_slice.start + T - 1] = 1.0 / efficiency

    A_ub = np.zeros((T, num_vars), dtype=np.float64)
    b_ub = u.copy()
    for t in range(T):
        A_ub[t, c_in_slice.start + t] = -1.0
        A_ub[t, c_out_slice.start + t] = 1.0

    bounds = [(0.0, Q)] * T + [(0.0, C)] * T + [(0.0, D)] * T

    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Failed to solve the battery-scheduling LP: {result.message}"
        )

    q = np.asarray(result.x[q_slice], dtype=np.float64)
    c_in = np.asarray(result.x[c_in_slice], dtype=np.float64)
    c_out = np.asarray(result.x[c_out_slice], dtype=np.float64)
    return q, c_in, c_out


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    """Solve the battery-scheduling problem with a linear-program formulation."""
    T, p, u, _, _, _ = _parse_problem(problem)
    q, c_in, c_out = _solve_lp(problem)
    c_net = c_in - c_out

    cost_without_battery = float(p @ u)
    cost_with_battery = float(p @ (u + c_net))
    savings = cost_without_battery - cost_with_battery

    return {
        "status": "optimal",
        "optimal": True,
        "battery_results": [
            {
                "q": q.tolist(),
                "c": c_net.tolist(),
                "c_in": c_in.tolist(),
                "c_out": c_out.tolist(),
                "cost": cost_with_battery,
            }
        ],
        "total_charging": c_net.tolist(),
        "cost_without_battery": cost_without_battery,
        "cost_with_battery": cost_with_battery,
        "savings": savings,
        "savings_percent": float(100.0 * savings / cost_without_battery),
    }


def validate_solution(problem: dict[str, Any], solution: Any) -> dict[str, float]:
    """Validate one candidate schedule and return cost and feasibility diagnostics."""
    if not isinstance(solution, dict):
        raise TypeError(f"Expected dict solution, got {type(solution).__name__}.")
    if not solution.get("optimal", False):
        raise ValueError("Solution must be marked as optimal.")

    required_keys = {
        "battery_results",
        "total_charging",
        "cost_without_battery",
        "cost_with_battery",
        "savings",
    }
    missing = required_keys.difference(solution)
    if missing:
        raise ValueError(f"Solution is missing keys: {', '.join(sorted(missing))}.")

    T, p, u, battery, deg_cost, num_batteries = _parse_problem(problem)
    battery_results = solution["battery_results"]
    if not isinstance(battery_results, list) or len(battery_results) != num_batteries:
        raise ValueError(f"Expected {num_batteries} battery result entries.")

    total_c = np.asarray(solution["total_charging"], dtype=np.float64)
    if total_c.shape != (T,) or not np.all(np.isfinite(total_c)):
        raise ValueError("total_charging must be a finite vector of length T.")

    q = np.asarray(battery_results[0]["q"], dtype=np.float64)
    c = np.asarray(battery_results[0]["c"], dtype=np.float64)
    c_in = np.asarray(battery_results[0]["c_in"], dtype=np.float64)
    c_out = np.asarray(battery_results[0]["c_out"], dtype=np.float64)

    for name, arr in (("q", q), ("c", c), ("c_in", c_in), ("c_out", c_out)):
        if arr.shape != (T,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be a finite vector of length T.")

    if not np.allclose(total_c, c, atol=FEASIBILITY_ATOL, rtol=0.0):
        raise ValueError("total_charging does not match the battery charging profile.")
    if not np.allclose(c, c_in - c_out, atol=FEASIBILITY_ATOL, rtol=0.0):
        raise ValueError("Net charging does not match c_in - c_out.")

    Q = battery["Q"]
    C = battery["C"]
    D = battery["D"]
    efficiency = battery["efficiency"]

    max_capacity_violation = float(max(max(-np.min(q), 0.0), max(np.max(q - Q), 0.0)))
    if max_capacity_violation > FEASIBILITY_ATOL:
        raise ValueError("Battery capacity constraints are violated.")

    max_charge_violation = float(
        max(max(-np.min(c_in), 0.0), max(np.max(c_in - C), 0.0))
    )
    max_discharge_violation = float(
        max(max(-np.min(c_out), 0.0), max(np.max(c_out - D), 0.0))
    )
    if (
        max_charge_violation > FEASIBILITY_ATOL
        or max_discharge_violation > FEASIBILITY_ATOL
    ):
        raise ValueError("Charge/discharge rate constraints are violated.")

    max_dynamics_residual = 0.0
    for t in range(T - 1):
        residual = abs(
            q[t + 1] - q[t] - efficiency * c_in[t] + (1.0 / efficiency) * c_out[t]
        )
        max_dynamics_residual = max(max_dynamics_residual, float(residual))
    cyclic_residual = abs(
        q[0] - q[T - 1] - efficiency * c_in[T - 1] + (1.0 / efficiency) * c_out[T - 1]
    )
    max_dynamics_residual = max(max_dynamics_residual, float(cyclic_residual))
    if max_dynamics_residual > FEASIBILITY_ATOL:
        raise ValueError("Battery dynamics or cyclic constraint is violated.")

    min_grid_margin = float(np.min(u + total_c))
    if min_grid_margin < -FEASIBILITY_ATOL:
        raise ValueError("No-power-back-to-grid constraint is violated.")

    cost_without_battery = float(solution["cost_without_battery"])
    cost_with_battery = float(solution["cost_with_battery"])
    savings = float(solution["savings"])

    expected_cost_without = float(p @ u)
    if abs(cost_without_battery - expected_cost_without) > COST_ATOL * (
        1.0 + abs(expected_cost_without)
    ):
        raise ValueError("cost_without_battery is inconsistent with p and u.")

    expected_cost_with = float(p @ (u + total_c))
    if deg_cost > 0.0:
        expected_cost_with += deg_cost * float(np.sum(c_in + c_out))
    if abs(cost_with_battery - expected_cost_with) > COST_ATOL * (
        1.0 + abs(expected_cost_with)
    ):
        raise ValueError(
            "cost_with_battery is inconsistent with the returned charging profile."
        )

    expected_savings = cost_without_battery - cost_with_battery
    if abs(savings - expected_savings) > COST_ATOL * (1.0 + abs(expected_savings)):
        raise ValueError("savings is inconsistent with the reported costs.")

    reference = solve_problem(problem)
    reference_cost = float(reference["cost_with_battery"])
    cost_gap = cost_with_battery - reference_cost
    if cost_gap > COST_ATOL + COST_RTOL * (1.0 + abs(reference_cost)):
        raise ValueError(
            f"Schedule is suboptimal by {cost_gap:.3e} relative to the reference optimum."
        )

    return {
        "cost_gap": float(cost_gap),
        "min_grid_margin": min_grid_margin,
        "max_dynamics_residual": float(max_dynamics_residual),
    }


def is_solution(problem: dict[str, Any], solution: Any) -> bool:
    """Return whether ``solution`` satisfies the adapted validator."""
    try:
        validate_solution(problem, solution)
    except Exception:
        return False
    return True
