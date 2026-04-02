from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog


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
    G = np.asarray(problem["G"], dtype=np.float64)
    sigma = np.asarray(problem["sigma"], dtype=np.float64)
    P_min = np.asarray(problem["P_min"], dtype=np.float64)
    P_max = np.asarray(problem["P_max"], dtype=np.float64)
    S_min = float(problem["S_min"])
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


def entrypoint(context: dict[str, object]) -> list[dict[str, object]]:
    """Exact baseline solver for the fixed batch of power-control problems."""
    outputs: list[dict[str, object]] = []
    for case in context["cases"]:
        problem = generate_problem(
            n=int(case["n"]),
            random_seed=int(case["random_seed"]),
        )
        outputs.append(solve_problem(problem))
    return outputs
