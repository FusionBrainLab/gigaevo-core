from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog


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

    return {
        "T": T,
        "p": p.tolist(),
        "u": u.tolist(),
        "batteries": [
            {
                "Q": 25.0,
                "C": 4.0,
                "D": 4.0,
                "efficiency": 0.9,
            }
        ],
        "deg_cost": 0.0,
        "num_batteries": 1,
    }


def _parse_problem(
    problem: dict[str, Any],
) -> tuple[int, np.ndarray, np.ndarray, dict[str, float]]:
    T = int(problem["T"])
    p = np.asarray(problem["p"], dtype=np.float64)
    u = np.asarray(problem["u"], dtype=np.float64)
    battery = problem["batteries"][0]
    return (
        T,
        p,
        u,
        {
            "Q": float(battery["Q"]),
            "C": float(battery["C"]),
            "D": float(battery["D"]),
            "efficiency": float(battery["efficiency"]),
        },
    )


def _solve_lp(problem: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, p, u, battery = _parse_problem(problem)
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
    A_eq[T - 1, q_slice.start] = 1.0
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
    _, p, u, _ = _parse_problem(problem)
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


def entrypoint(context: dict[str, object]) -> list[dict[str, object]]:
    """Exact baseline solver for the fixed batch of battery-scheduling problems."""
    outputs: list[dict[str, object]] = []
    for case in context["cases"]:
        problem = generate_problem(
            n=int(case["n"]),
            random_seed=int(case["random_seed"]),
        )
        outputs.append(solve_problem(problem))
    return outputs
