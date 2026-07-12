"""Paired per-sample score comparison: shared statistic + Program accessors.

When ``validate()`` emits a per-sample score vector on a fixed eval set,
two programs from the same run can be compared PAIRED on shared samples
instead of via two noisy scalar means. Consumers (archive selectors, card
crediting) own orientation and acceptance thresholds; this module owns the
statistic and the ``program.metadata`` contract so every consumer applies
the same fallback predicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from loguru import logger
import numpy as np

from gigaevo.memory.cards import Measurement
from gigaevo.programs.program import Program

PER_SAMPLE_SCORES_KEY = "per_sample_scores"

# Float slack for mean(per_sample_scores) == metrics[key]; a larger gap means
# the vector no longer describes the gated metric (silent contract drift).
COHERENCE_TOL = 1e-4


@runtime_checkable
class PairedComparison(Protocol):
    """Strategy seam for paired-vector comparison statistics."""

    def probability_better(
        self, challenger: np.ndarray, incumbent: np.ndarray
    ) -> float:
        """P(challenger beats incumbent) in [0, 1], paired on shared samples.

        Implementations must be pure (same inputs -> same output) and
        order-symmetric: ``p(a, b) + p(b, a) == 1``.
        """
        ...

    def estimate(self, challenger: np.ndarray, incumbent: np.ndarray) -> Measurement:
        """Paired effect estimate: mean(challenger - incumbent) with its se.

        Same purity/antisymmetry contract: ``estimate(a, b).value ==
        -estimate(b, a).value`` and the se is order-invariant.
        """
        ...


@dataclass(frozen=True)
class PairedBootstrap:
    """Bootstrap P(mean(challenger - incumbent) > 0) over paired resamples."""

    n_resamples: int = 2000
    seed: int = 0

    def probability_better(
        self, challenger: np.ndarray, incumbent: np.ndarray
    ) -> float:
        diff = np.asarray(challenger, dtype=float) - np.asarray(incumbent, dtype=float)
        # Fresh fixed-seed rng per call: (a,b) and (b,a) resample the same
        # indices, so verdicts are order-independent and exactly complementary.
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, diff.size, size=(self.n_resamples, diff.size))
        means = diff[idx].mean(axis=1)
        # Ties count half so identical vectors land at exactly 0.5.
        return float((means > 0).mean() + 0.5 * (means == 0).mean())

    def estimate(self, challenger: np.ndarray, incumbent: np.ndarray) -> Measurement:
        diff = np.asarray(challenger, dtype=float) - np.asarray(incumbent, dtype=float)
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, diff.size, size=(self.n_resamples, diff.size))
        means = diff[idx].mean(axis=1)
        # Value is the exact paired mean; the bootstrap only prices its spread.
        return Measurement(value=float(diff.mean()), se=float(means.std(ddof=1)))


def get_per_sample_scores(
    program: Program, *, metric_key: str | None = None
) -> np.ndarray | None:
    """Return the program's per-sample score vector, or None when unusable.

    With ``metric_key`` the vector must also cohere with the stored metric
    (mean within COHERENCE_TOL) — consumers gating on that metric must not
    trust a vector that describes something else.
    """
    raw: Any = program.metadata.get(PER_SAMPLE_SCORES_KEY)
    if raw is None:
        return None
    try:
        scores = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        logger.warning(
            "[per_sample_scores] program {} carries a non-numeric vector — ignored",
            program.id,
        )
        return None
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        logger.warning(
            "[per_sample_scores] program {} vector unusable "
            "(ndim={}, size={}, finite={}) — ignored",
            program.id,
            scores.ndim,
            scores.size,
            bool(np.isfinite(scores).all()) if scores.size else False,
        )
        return None
    if metric_key is not None:
        metric = program.metrics.get(metric_key)
        if metric is None or abs(float(scores.mean()) - float(metric)) > COHERENCE_TOL:
            logger.warning(
                "[per_sample_scores] program {} vector incoherent with "
                "metrics[{!r}] (mean={:.6f}, metric={}) — ignored",
                program.id,
                metric_key,
                float(scores.mean()),
                metric,
            )
            return None
    return scores


def get_paired_scores(
    a: Program, b: Program, *, metric_key: str | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return both programs' vectors when a paired comparison is valid.

    None unless both vectors exist, are usable, cohere with ``metric_key``
    (when given), and have equal length (same fixed eval set).
    """
    scores_a = get_per_sample_scores(a, metric_key=metric_key)
    scores_b = get_per_sample_scores(b, metric_key=metric_key)
    if scores_a is None or scores_b is None or scores_a.shape != scores_b.shape:
        return None
    return scores_a, scores_b
