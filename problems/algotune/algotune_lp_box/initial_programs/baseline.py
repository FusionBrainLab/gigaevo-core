from __future__ import annotations

from algotune_lp_box_helper import generate_problem, solve_problem


def entrypoint(context: dict[str, object]) -> list[dict[str, object]]:
    """Exact baseline solver for the fixed batch of boxed linear programs."""
    outputs: list[dict[str, object]] = []
    for case in context["cases"]:
        problem = generate_problem(
            n=int(case["n"]),
            random_seed=int(case["random_seed"]),
        )
        outputs.append(solve_problem(problem))
    return outputs
