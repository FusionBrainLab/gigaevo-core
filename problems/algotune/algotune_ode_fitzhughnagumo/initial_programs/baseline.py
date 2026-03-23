from __future__ import annotations

from typing import Any

from algotune_ode_fitzhughnagumo_helper import generate_problem, solve_problem


def entrypoint(context: dict[str, Any]) -> list[list[float]]:
    """Exact baseline solver using SciPy's RK45 integrator."""
    outputs: list[list[float]] = []
    for case in context["cases"]:
        problem = generate_problem(n=int(case["n"]), random_seed=int(case["random_seed"]))
        outputs.append(solve_problem(problem))
    return outputs
