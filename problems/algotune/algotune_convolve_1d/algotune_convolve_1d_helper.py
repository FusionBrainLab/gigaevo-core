"""Helper functions for the adapted AlgoTune ``convolve_1d`` task.

This mirrors the core problem generation and validation logic from
``AlgoTune-main/AlgoTuneTasks/convolve_1d/convolve_1d.py`` without importing
the full AlgoTune runtime, which depends on optional packages that are not
installed in this environment.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy import signal

MODE = "full"
TOLERANCE = 1.0e-6


class CaseSpec(TypedDict):
    n: int
    random_seed: int


def get_case_specs() -> list[CaseSpec]:
    """Deterministic evaluation suite spanning several problem scales."""
    return [
        {"n": 1, "random_seed": 1234},
        {"n": 2, "random_seed": 2024},
        {"n": 4, "random_seed": 7},
        {"n": 6, "random_seed": 99},
        {"n": 8, "random_seed": 31415},
    ]


def generate_problem(
    n: int = 1, random_seed: int = 1234
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one convolve_1d instance using the same shapes as AlgoTune."""
    rng = np.random.default_rng(random_seed)
    a = rng.standard_normal(30 * n)
    b = rng.standard_normal(8 * n)
    return a, b


def solve_problem(problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Reference solver mirroring AlgoTune's task implementation."""
    a, b = problem
    return signal.convolve(a, b, mode=MODE)


def relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Relative L2 error used by the adapted validator."""
    return float(
        np.linalg.norm(candidate - reference) / (np.linalg.norm(reference) + 1.0e-12)
    )


def is_solution(problem: tuple[np.ndarray, np.ndarray], solution: np.ndarray) -> bool:
    """Match AlgoTune's correctness check for ``convolve_1d``."""
    reference = solve_problem(problem)
    error = relative_error(reference, solution)
    return error <= TOLERANCE
