"""Independent verifier for the spherical-codes benchmark.

The objective, mu(X) = max_{i<j} <x_i, x_j>, is a one-line formula on a Gram
matrix; there is no second way to compute it and no room for a bug that a
reimplementation would not also make. So the useful independence here is not in
the objective but in the *feasibility contract*, which is where an improver can
actually cheat: return the wrong shape, return rows that are not unit vectors
(shrinking every row shrinks every inner product), or slip in a NaN that makes
`max` lie. This module re-checks all of that without importing the grader.

Tolerances are lifted from the existing grader rather than re-derived:
`problems/spherical_codes_improver_general/validate.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Row norms must be within this of 1. From SPHERICAL_NORM_TOL's default in
# validate.py::_Cfg — tight, because float64 renormalization lands far inside it
# and anything looser is an improver quietly buying mu by deflating its rows.
NORM_ABS_TOLERANCE = 1e-12

# The paper's grader accepts on `mu <= current_mu` (validate.py, stages A and B):
# non-strict, so an equal-mu candidate replaces the incumbent. That lateral move
# is deliberate — it is how the search traverses the plateau these packings sit
# on — and it is kept here rather than tightened, so the controller reproduces
# the accept rule the published numbers were produced under.
ACCEPTANCE_ABS_TOLERANCE = 0.0


@dataclass(frozen=True)
class SphericalReport:
    feasible: bool
    mu: float
    max_norm_error: float
    reason: str | None = None


def mu_of(points: np.ndarray) -> float:
    """Signed maximum pairwise inner product; lower is better."""
    gram = points @ points.T
    np.fill_diagonal(gram, -np.inf)
    return float(gram.max())


def verify(config, dimension: int, count: int) -> SphericalReport:
    try:
        points = np.asarray(config, dtype=np.float64)
    except (TypeError, ValueError) as err:
        return SphericalReport(False, np.inf, np.inf, f"unreadable config: {err}")

    if points.shape != (count, dimension):
        return SphericalReport(
            False,
            np.inf,
            np.inf,
            f"expected shape ({count}, {dimension}), got {points.shape}",
        )
    if not np.all(np.isfinite(points)):
        return SphericalReport(False, np.inf, np.inf, "non-finite coordinates")

    norm_error = float(np.abs(np.linalg.norm(points, axis=1) - 1.0).max())
    if norm_error > NORM_ABS_TOLERANCE:
        return SphericalReport(
            False,
            np.inf,
            norm_error,
            f"rows are not unit vectors: worst norm error {norm_error:.3e}",
        )

    return SphericalReport(True, mu_of(points), norm_error)


@dataclass(frozen=True)
class SphericalBenchmark:
    """The controller's view of one (d, N) configuration.

    One benchmark instance per configuration: the experiment's headline number is
    the mean relative gain over Cohn across a *set* of configurations, but that
    aggregation belongs to the study, not to the thing the controller optimizes.
    """

    dimension: int
    count: int

    def validate(self, config) -> float | None:
        report = verify(config, self.dimension, self.count)
        return report.mu if report.feasible else None

    def better(self, candidate: float, incumbent: float) -> bool:
        return candidate <= incumbent - ACCEPTANCE_ABS_TOLERANCE
