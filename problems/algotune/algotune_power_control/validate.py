from __future__ import annotations

from typing import Any

from algotune_power_control_helper import generate_problem, validate_solution


def validate(context: dict[str, Any], outputs: list[Any]) -> dict[str, float]:
    """Validate a batch of power-control outputs against the fixed suite."""
    cases = context.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Context must contain a non-empty 'cases' list.")
    if not isinstance(outputs, list):
        raise TypeError(f"Expected list of outputs, got {type(outputs).__name__}.")
    if len(outputs) != len(cases):
        raise ValueError(f"Expected {len(cases)} outputs, got {len(outputs)}.")

    solved = 0

    for idx, (case, output) in enumerate(zip(cases, outputs)):
        problem = generate_problem(
            n=int(case["n"]),
            random_seed=int(case["random_seed"]),
        )
        try:
            validate_solution(problem, output)
        except Exception as exc:
            raise ValueError(f"Case {idx} is invalid: {exc}") from exc
        solved += 1

    exact_case_fraction = float(solved / len(cases))

    if solved != len(cases):
        raise ValueError(
            f"Output is not exact enough for all cases: matched {solved} / {len(cases)} cases."
        )

    return {
        "exact_case_fraction": exact_case_fraction,
        "is_valid": 1.0,
    }
