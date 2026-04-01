from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

STABILITY_TOLERANCE = 1.0e-10
GEOMETRIC_SERIES_TOLERANCE = 1.0e-12
GEOMETRIC_SERIES_MAX_ITERATIONS = 10_000


def generate_problem(
    n: int = 1, random_seed: int = 1234
) -> dict[str, list[list[float]]]:
    rng = np.random.default_rng(random_seed)
    A = rng.normal(0, 1, (n, n))

    if rng.random() < 0.7:
        eigenvalues = np.linalg.eigvals(A)
        max_magnitude = float(np.max(np.abs(eigenvalues)))
        A = A * (0.8 / max_magnitude)

    return {"A": A.tolist()}


def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def is_stable_matrix(A: np.ndarray) -> bool:
    return spectral_radius(A) < 1.0 - STABILITY_TOLERANCE


def solve_problem(problem: dict[str, Any]) -> dict[str, Any]:
    A = np.asarray(problem["A"], dtype=np.float64)
    if not is_stable_matrix(A):
        return {"is_stable": False, "P": None}

    P = solve_lyapunov_certificate(A)
    return {"is_stable": True, "P": P.tolist()}


def solve_lyapunov_certificate(A: np.ndarray) -> np.ndarray:
    identity = np.eye(A.shape[0], dtype=np.float64)

    try:
        P = solve_discrete_lyapunov(A.T, identity)
        return 0.5 * (P + P.T)
    except Exception:
        pass

    P = identity.copy()
    term = identity.copy()
    for _ in range(GEOMETRIC_SERIES_MAX_ITERATIONS):
        term = A.T @ term @ A
        P = P + term
        if np.linalg.norm(term, ord="fro") <= GEOMETRIC_SERIES_TOLERANCE * (
            1.0 + np.linalg.norm(P, ord="fro")
        ):
            return 0.5 * (P + P.T)

    raise RuntimeError("Failed to compute a Lyapunov certificate for a stable system.")


def entrypoint(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Exact baseline solver using SciPy's discrete Lyapunov equation solver."""
    outputs: list[dict[str, Any]] = []
    for case in context["cases"]:
        problem = generate_problem(
            n=int(case["n"]),
            random_seed=int(case["random_seed"]),
        )
        outputs.append(solve_problem(problem))
    return outputs
