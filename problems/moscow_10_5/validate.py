"""Validator for a counterexample search for the real Moscow problem."""

from __future__ import annotations

from helper import NUM_SUBSETS, analyze_candidate
import numpy as np


def validate(candidate: object):
    """Exactly evaluate all C(10, 5) = 252 square row submatrices.

    Fitness is ``1 - Phi``, where

        Phi = 10 * max_S sigma_min(Q[S, :])**2

    and Q is an orthonormal basis for the returned column space.  A genuine
    counterexample has positive fitness.  Equality examples have fitness zero.
    """

    analysis = analyze_candidate(candidate)

    metrics = {
        "fitness": analysis.fitness,
        "phi": analysis.phi,
        "max_min_eigenvalue": analysis.max_min_eigenvalue,
        "basis_density": analysis.basis_density,
        "numerical_basis_count": float(analysis.numerical_basis_count),
        "active_log_density": analysis.active_log_density,
        "active_count": float(analysis.active_count),
        "leverage_cv": analysis.leverage_cv,
        "leverage_min": analysis.leverage_min,
        "leverage_max": analysis.leverage_max,
        "is_valid": 1.0,
    }

    top_indices = np.argsort(analysis.subset_scores)[-5:][::-1]
    top_values = [float(10.0 * analysis.subset_scores[index]) for index in top_indices]
    one_based_subset = tuple(index + 1 for index in analysis.best_subset)
    verdict = (
        "COUNTEREXAMPLE CANDIDATE"
        if analysis.fitness > 1.0e-10
        else "no counterexample"
    )
    feedback = (
        f"{verdict}: Phi={analysis.phi:.12g}, "
        f"fitness=1-Phi={analysis.fitness:.12g}. "
        f"Best row subset (1-based)={one_based_subset}; "
        f"top five normalized subset values={top_values}. "
        f"Near-active subsets={analysis.active_count}/{NUM_SUBSETS}; "
        f"numerical bases={analysis.numerical_basis_count}/{NUM_SUBSETS}; "
        f"leverage range=[{analysis.leverage_min:.6g}, "
        f"{analysis.leverage_max:.6g}], CV={analysis.leverage_cv:.6g}."
    )
    return metrics, feedback
