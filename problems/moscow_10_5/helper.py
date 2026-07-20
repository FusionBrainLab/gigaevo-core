"""Exact numerical helpers for the real Moscow problem at (n, r) = (10, 5).

The candidate is represented by any full-column-rank 10 x 5 matrix.  Only its
column space matters: ``orthonormal_basis`` replaces it by an orthonormal basis
before any score is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

N_ROWS = 10
RANK = 5
NUM_SUBSETS = 252
SUBSETS = np.asarray(
    list(combinations(range(N_ROWS), RANK)),
    dtype=np.intp,
)

# These tolerances define the two MAP-Elites behavior descriptors.
RELATIVE_RANK_TOL = 1.0e-12
NUMERICAL_BASIS_EIGENVALUE_TOL = 1.0e-10
NEAR_ACTIVE_FRACTION = 0.95


@dataclass(frozen=True)
class MoscowAnalysis:
    """All exact-enumeration statistics used by ``validate.py``."""

    basis: np.ndarray
    projector: np.ndarray
    subset_scores: np.ndarray
    best_subset_index: int
    best_subset: tuple[int, ...]
    max_min_eigenvalue: float
    phi: float
    fitness: float
    numerical_basis_count: int
    basis_density: float
    active_count: int
    active_log_density: float
    leverage_min: float
    leverage_max: float
    leverage_cv: float


def orthonormal_basis(candidate: object) -> np.ndarray:
    """Return a stable orthonormal basis for a valid candidate's column space.

    Scaling and right multiplication by an invertible 5 x 5 matrix therefore
    do not affect the score.  Extremely ill-conditioned representations are
    rejected because their column space cannot be recovered reliably in
    float64 arithmetic.
    """

    try:
        raw_matrix = np.asarray(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate must be convertible to a numeric array.") from exc
    if np.iscomplexobj(raw_matrix):
        raise ValueError("Candidate must be real; complex entries are not allowed.")
    try:
        matrix = np.asarray(raw_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate must be convertible to a numeric array.") from exc

    if matrix.shape != (N_ROWS, RANK):
        raise ValueError(
            f"Expected a ({N_ROWS}, {RANK}) matrix, got shape {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Candidate contains NaN or infinite entries.")

    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        raise ValueError("Candidate is the zero matrix, not a rank-5 subspace.")

    # Rescaling first avoids overflow/underflow while preserving the subspace.
    normalized = matrix / scale
    try:
        left_vectors, singular_values, _ = np.linalg.svd(
            normalized,
            full_matrices=False,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("SVD failed while extracting the candidate subspace.") from exc

    relative_smallest = float(singular_values[-1] / singular_values[0])
    if relative_smallest <= RELATIVE_RANK_TOL:
        raise ValueError(
            "Candidate is rank deficient or numerically unrecoverable: "
            f"relative smallest singular value {relative_smallest:.3e} "
            f"must exceed {RELATIVE_RANK_TOL:.1e}."
        )
    return left_vectors[:, :RANK]


def subset_min_eigenvalues_from_basis(basis: np.ndarray) -> np.ndarray:
    """Evaluate all 252 row subsets for an orthonormal 10 x 5 basis.

    For a subset S, the returned value is

        lambda_min(P[S,S]) = sigma_min(basis[S,:])**2,

    where P is the rank-5 orthogonal projector onto the candidate subspace.
    """

    basis = np.asarray(basis, dtype=np.float64)
    if basis.shape != (N_ROWS, RANK):
        raise ValueError(
            f"Expected an orthonormal ({N_ROWS}, {RANK}) basis, "
            f"got shape {basis.shape}."
        )

    projector = basis @ basis.T
    principal_blocks = projector[
        SUBSETS[:, :, None],
        SUBSETS[:, None, :],
    ]
    scores = np.linalg.eigvalsh(principal_blocks)[:, 0]

    # Mathematically every score lies in [0, 1].  Clipping only removes
    # roundoff such as -3e-16 on exactly singular graph-matroid subsets.
    return np.clip(scores, 0.0, 1.0)


def analyze_candidate(candidate: object) -> MoscowAnalysis:
    """Canonicalize and exactly score one candidate."""

    basis = orthonormal_basis(candidate)
    projector = basis @ basis.T
    subset_scores = subset_min_eigenvalues_from_basis(basis)

    best_subset_index = int(np.argmax(subset_scores))
    max_min_eigenvalue = float(subset_scores[best_subset_index])
    phi = float(np.clip(N_ROWS * max_min_eigenvalue, 0.0, float(N_ROWS)))
    fitness = float(np.clip(1.0 - phi, 1.0 - N_ROWS, 1.0))

    numerical_basis_count = int(
        np.count_nonzero(
            subset_scores > NUMERICAL_BASIS_EIGENVALUE_TOL,
        )
    )
    basis_density = float(numerical_basis_count / NUM_SUBSETS)

    active_threshold = NEAR_ACTIVE_FRACTION * max_min_eigenvalue
    active_count = int(np.count_nonzero(subset_scores >= active_threshold))
    # active_count is at least one.  The logarithm spreads the small counts
    # (1, 2, 3, ...) that otherwise collapse into a single linear bin.
    active_log_density = float(
        np.clip(
            np.log(active_count) / np.log(NUM_SUBSETS),
            0.0,
            1.0,
        )
    )

    leverages = np.clip(np.diag(projector), 0.0, 1.0)
    leverage_mean = RANK / N_ROWS
    leverage_cv = float(np.clip(np.std(leverages) / leverage_mean, 0.0, 1.0))

    return MoscowAnalysis(
        basis=basis,
        projector=projector,
        subset_scores=subset_scores,
        best_subset_index=best_subset_index,
        best_subset=tuple(int(i) for i in SUBSETS[best_subset_index]),
        max_min_eigenvalue=max_min_eigenvalue,
        phi=phi,
        fitness=fitness,
        numerical_basis_count=numerical_basis_count,
        basis_density=basis_density,
        active_count=active_count,
        active_log_density=active_log_density,
        leverage_min=float(np.min(leverages)),
        leverage_max=float(np.max(leverages)),
        leverage_cv=leverage_cv,
    )
