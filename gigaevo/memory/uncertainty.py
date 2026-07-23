"""Shared uncertainty calculation for memory outcome deltas."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_EVALUATION_MEASUREMENTS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORE_SIGNATURE_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORES_METADATA_KEY,
)
from gigaevo.programs.metrics.evaluation import (
    EVALUATION_MEASUREMENTS_METADATA_KEY,
    reported_standard_error,
)
from gigaevo.programs.metrics.paired import (
    COHERENCE_TOL,
    PER_SAMPLE_SCORES_KEY,
    PER_SAMPLE_SIGNATURE_KEY,
)

if TYPE_CHECKING:
    from gigaevo.programs.program import Program


def _paired_uncertainty(
    program: Program,
    *,
    child_fitness: float,
    base_fitness: float,
    higher_is_better: bool,
) -> tuple[float | None, int | None, Literal["scalar", "paired"], str]:
    """Analytic paired SE only when the ordered evaluation cohorts match."""

    child_raw = program.get_metadata(PER_SAMPLE_SCORES_KEY)
    base_raw = program.get_metadata(MUTATION_MEMORY_BASE_SCORES_METADATA_KEY)
    child_signature = program.get_metadata(PER_SAMPLE_SIGNATURE_KEY)
    base_signature = program.get_metadata(
        MUTATION_MEMORY_BASE_SCORE_SIGNATURE_METADATA_KEY
    )
    if (
        not isinstance(child_signature, str)
        or not child_signature
        or child_signature != base_signature
    ):
        return None, None, "scalar", ""
    try:
        child = np.asarray(child_raw, dtype=float)
        base = np.asarray(base_raw, dtype=float)
    except (TypeError, ValueError):
        return None, None, "scalar", ""
    if (
        child.ndim != 1
        or base.ndim != 1
        or child.shape != base.shape
        or child.size < 2
        or not np.isfinite(child).all()
        or not np.isfinite(base).all()
        or abs(float(child.mean()) - child_fitness) > COHERENCE_TOL
        or abs(float(base.mean()) - base_fitness) > COHERENCE_TOL
    ):
        return None, None, "scalar", ""
    differences = child - base if higher_is_better else base - child
    se = float(np.std(differences, ddof=1) / np.sqrt(child.size))
    if not np.isfinite(se) or se < 0.0:
        return None, None, "scalar", ""
    return se, int(child.size), "paired", child_signature


def _reported_uncertainty(
    program: Program,
    *,
    metric_key: str,
    child_fitness: float,
    base_fitness: float,
) -> float | None:
    """Independent-difference SE from two evaluator-reported measurements."""

    child_se = reported_standard_error(
        program.get_metadata(EVALUATION_MEASUREMENTS_METADATA_KEY),
        metric_key=metric_key,
        expected_value=child_fitness,
    )
    base_se = reported_standard_error(
        program.get_metadata(MUTATION_MEMORY_BASE_EVALUATION_MEASUREMENTS_METADATA_KEY),
        metric_key=metric_key,
        expected_value=base_fitness,
    )
    if child_se is None or base_se is None:
        return None
    combined = math.hypot(child_se, base_se)
    return combined if math.isfinite(combined) else None


def outcome_uncertainty(
    program: Program,
    *,
    metric_key: str,
    child_fitness: float,
    base_fitness: float,
    higher_is_better: bool,
) -> tuple[float | None, int | None, Literal["scalar", "paired"], str]:
    """Prefer paired vectors, then use reported marginal standard errors."""

    paired = _paired_uncertainty(
        program,
        child_fitness=child_fitness,
        base_fitness=base_fitness,
        higher_is_better=higher_is_better,
    )
    if paired[0] is not None:
        return paired
    reported = _reported_uncertainty(
        program,
        metric_key=metric_key,
        child_fitness=child_fitness,
        base_fitness=base_fitness,
    )
    return reported, None, "scalar", ""
