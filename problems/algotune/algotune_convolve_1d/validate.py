from __future__ import annotations

from typing import Any

from algotune_convolve_1d_helper import (
    generate_problem,
    is_solution,
    relative_error,
    solve_problem,
)
import numpy as np


def validate(context: dict[str, Any], outputs: list[Any]) -> dict[str, float]:
    """Validate a batch of convolution outputs against the fixed case suite."""
    cases = context.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Context must contain a non-empty 'cases' list.")
    if not isinstance(outputs, list):
        raise TypeError(f"Expected list of outputs, got {type(outputs).__name__}.")
    if len(outputs) != len(cases):
        raise ValueError(f"Expected {len(cases)} outputs, got {len(outputs)}.")

    solved = 0
    errors: list[float] = []

    for idx, (case, output) in enumerate(zip(cases, outputs)):
        problem = generate_problem(
            n=int(case["n"]),
            random_seed=int(case["random_seed"]),
        )
        reference = solve_problem(problem)
        candidate = np.asarray(output, dtype=np.float64)

        if candidate.shape != reference.shape:
            raise ValueError(
                f"Case {idx} has wrong shape: expected {reference.shape}, got {candidate.shape}."
            )
        if not np.all(np.isfinite(candidate)):
            raise ValueError(f"Case {idx} contains non-finite values.")

        err = relative_error(reference, candidate)
        errors.append(err)
        if is_solution(problem, candidate):
            solved += 1

    exact_case_fraction = float(solved / len(cases))
    avg_error = float(np.mean(errors))

    if solved != len(cases):
        raise ValueError(
            f"Output is not exact enough for all cases: matched {solved} / {len(cases)} cases."
        )

    return {
        "avg_relative_error": avg_error,
        "exact_case_fraction": exact_case_fraction,
        "is_valid": 1.0,
    }
