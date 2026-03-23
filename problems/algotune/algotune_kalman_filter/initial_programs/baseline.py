from __future__ import annotations

from typing import Any

from algotune_kalman_filter_helper import generate_problem, solve_problem


def entrypoint(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Exact baseline solver using a closed-form least-squares reduction."""
    outputs: list[dict[str, Any]] = []
    for case in context["cases"]:
        problem = generate_problem(
            n=int(case["n"]),
            random_seed=int(case["random_seed"]),
        )
        outputs.append(solve_problem(problem))
    return outputs
