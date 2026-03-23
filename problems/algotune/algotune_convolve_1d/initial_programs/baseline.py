from __future__ import annotations

from typing import Any

import numpy as np

from algotune_convolve_1d_helper import generate_problem


def entrypoint(context: dict[str, Any]) -> list[np.ndarray]:
    """Exact baseline solver using NumPy's full 1D convolution."""
    outputs: list[np.ndarray] = []
    for case in context["cases"]:
        a, b = generate_problem(
            n=int(case["n"]),
            random_seed=int(case["random_seed"]),
        )
        outputs.append(np.convolve(a, b, mode="full"))
    return outputs
