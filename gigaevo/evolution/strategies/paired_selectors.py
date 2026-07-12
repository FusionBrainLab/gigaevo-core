"""Noise-aware archive selectors gating on paired per-sample comparisons."""

from __future__ import annotations

from loguru import logger

from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.metrics.paired import (
    PairedBootstrap,
    PairedComparison,
    get_paired_scores,
)
from gigaevo.programs.program import Program


class PairedBootstrapArchiveSelector(SumArchiveSelector):
    """Archive replacement gated on P(paired mean diff > 0) >= p_accept.

    Requires both programs to carry ``metadata["per_sample_scores"]`` from the
    same fixed eval set (see ``gigaevo.programs.metrics.paired``). Whenever the
    paired path cannot run — missing/mismatched/incoherent vectors, or
    ``p_accept=0.5`` (the documented OFF position) — the decision delegates to
    ``SumArchiveSelector.__call__``, bit-identical to the point-comparison rule.

    Substitutable for ``SumArchiveSelector`` at every call site; construction
    is deliberately narrower (single fitness key only).
    """

    def __init__(
        self,
        *args,
        p_accept: float = 0.75,
        comparison: PairedComparison | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if len(self.fitness_keys) != 1:
            raise ValueError(
                "PairedBootstrapArchiveSelector is single-key; "
                "use SumArchiveSelector for weighted multi-key sums"
            )
        if not 0.5 <= p_accept < 1.0:
            raise ValueError(f"p_accept must be in [0.5, 1.0), got {p_accept}")
        if p_accept == 0.5 and comparison is not None:
            raise ValueError(
                "p_accept=0.5 is the OFF position and never consults a "
                "comparison; drop the injected comparison or raise p_accept"
            )
        self.p_accept = p_accept
        self._comparison = (
            None if p_accept == 0.5 else (comparison or PairedBootstrap())
        )

    def __call__(self, new: Program, current: Program) -> bool:
        if self._comparison is None:
            return super().__call__(new, current)
        key = self.fitness_keys[0]
        paired = get_paired_scores(new, current, metric_key=key)
        if paired is None:
            logger.debug(
                "[PairedBootstrapArchiveSelector] {} vs {} -> point-comparison "
                "fallback (missing/mismatched/incoherent per-sample scores)",
                new.id,
                current.id,
            )
            return super().__call__(new, current)
        new_scores, current_scores = paired
        if not self.fitness_key_higher_is_better[key]:
            new_scores, current_scores = -new_scores, -current_scores
        p_better = self._comparison.probability_better(new_scores, current_scores)
        result = p_better >= self.p_accept
        logger.debug(
            "[PairedBootstrapArchiveSelector] {} vs {} -> {} "
            "(P(better)={:.3f}, p_accept={}, key={})",
            new.id,
            current.id,
            "ACCEPT" if result else "REJECT",
            p_better,
            self.p_accept,
            key,
        )
        return result
