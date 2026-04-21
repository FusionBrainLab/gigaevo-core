"""Shared-benchmark filter for D-side LineageStage.

Rationale
---------
D's LineageStage narrates parent-D → child-D fitness transitions to the
mutation LLM. In adversarial co-evolution with a rotating G Hall-of-Fame
the raw whole-eval deltas are apples-to-oranges: parent and child are
usually scored against different G's. This stage subclasses LineageStage
and overrides preprocess() to:

  1) Filter out parents that share fewer than min_shared evaluation
     opponents with the child (tracker.faced_by_d intersection).
  2) Build HoF-invariant per-metric means over the shared opponents,
     respecting MetricsContext sentinels and the is_valid gate, and pass
     them to the LLM agent as TransitionEvidence.

Installation
------------
AdversarialAsymmetricPipelineBuilder uses PipelineBuilder.replace_stage(
"LineageStage", ...) on D runs only (see Task 7). Node name stays
"LineageStage" so downstream edges are preserved.

Future work
-----------
G-side analog deferred — see experiments/IDEAS.yaml:
heilbron-g-side-lineage-filter.
"""

from __future__ import annotations

from collections.abc import Collection
import math
from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.llm.agents.lineage import TransitionEvidence
from gigaevo.programs.core_types import ProgramStageResult, StageIO
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.stages.insights_lineage import LineageStage

if TYPE_CHECKING:
    from gigaevo.adversarial.dg_tracker import DGImprovementTracker


def _aggregate_shared_metrics(
    parent_by_g: dict[str, dict[str, float]],
    child_by_g: dict[str, dict[str, float]],
    shared_g: Collection[str],
    ctx: MetricsContext,
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Return (shared_parent_metrics, shared_child_metrics, per_metric_counts).

    Rules:
      - VALIDITY_KEY aggregated as a plain mean over ALL shared G's
        (0/1 indicator → mean is the valid-rate).
      - Other metrics: a G contributes only when ctx.is_valid holds on
        BOTH parent and child AND neither value is a sentinel per
        ctx.is_sentinel.
      - Zero surviving G's for a metric → both sides are set to that
        metric's sentinel value if one is declared, otherwise NaN.
        NaN propagates as a missing-value marker rather than a spurious
        0.0 the LLM could mistake for a real measurement.
    """
    shared_g = list(shared_g)

    all_metrics: set[str] = set()
    for g in shared_g:
        all_metrics |= parent_by_g.get(g, {}).keys()
        all_metrics |= child_by_g.get(g, {}).keys()

    shared_parent: dict[str, float] = {}
    shared_child: dict[str, float] = {}
    counts: dict[str, int] = {}

    for m in all_metrics:
        if m == VALIDITY_KEY:
            # Invariant: DGImprovementTracker.record_batch always writes
            # is_valid for every (D,G) pair, so .get(m, 0.0) is defensive
            # — missing key implies a pre-v4 row or a writer bypassing
            # record_batch, which the stage can safely treat as invalid.
            p_vals = [float(parent_by_g[g].get(m, 0.0)) for g in shared_g]
            c_vals = [float(child_by_g[g].get(m, 0.0)) for g in shared_g]
            shared_parent[m] = sum(p_vals) / len(p_vals) if p_vals else 0.0
            shared_child[m] = sum(c_vals) / len(c_vals) if c_vals else 0.0
            counts[m] = len(shared_g)
            continue

        p_vals, c_vals = [], []
        for g in shared_g:
            p = parent_by_g.get(g, {})
            c = child_by_g.get(g, {})
            if not ctx.is_valid(p) or not ctx.is_valid(c):
                continue
            pv, cv = p.get(m), c.get(m)
            if pv is None or cv is None:
                continue
            if ctx.is_sentinel(m, pv) or ctx.is_sentinel(m, cv):
                continue
            p_vals.append(float(pv))
            c_vals.append(float(cv))

        counts[m] = len(p_vals)
        if p_vals:
            shared_parent[m] = sum(p_vals) / len(p_vals)
            shared_child[m] = sum(c_vals) / len(c_vals)
        else:
            spec = ctx.specs.get(m)
            fallback = (
                spec.sentinel_value
                if spec and spec.sentinel_value is not None
                else math.nan
            )
            shared_parent[m] = fallback
            shared_child[m] = fallback

    return shared_parent, shared_child, counts


class SharedBenchmarkFilteredLineageStage(LineageStage):
    """LineageStage variant: filter parents by shared eval benchmark.

    A parent survives iff
    ``|tracker.faced_by_d(child) ∩ tracker.faced_by_d(parent)| >= min_shared``.
    For survivors, builds a TransitionEvidence from tracker.metrics_by_d
    with sentinel + is_valid gating and passes it to the LLM agent.
    """

    def __init__(
        self,
        *,
        tracker: DGImprovementTracker,
        metrics_context: MetricsContext,
        min_shared: int = 1,
        inject_shared_evidence: bool = True,
        **kwargs: Any,
    ):
        # min_shared < 1 would keep parents whose shared-opponent set is
        # empty and then hand the LLM an all-sentinel evidence block
        # (see _aggregate_shared_metrics zero-survivor branch). That
        # defeats the whole point of the HoF-invariant subset — reject
        # it loudly rather than silently degrading.
        if min_shared < 1:
            raise ValueError(
                f"min_shared must be >= 1 (got {min_shared}); use the base "
                "LineageStage for unfiltered lineage narratives."
            )
        super().__init__(metrics_context=metrics_context, **kwargs)
        self._tracker = tracker
        self._metrics_context = metrics_context
        self._min_shared = min_shared
        self._inject_shared_evidence = inject_shared_evidence

    async def preprocess(
        self, program: Program, params: StageIO
    ) -> dict[str, Any] | ProgramStageResult:
        parent_ids: list[str] = list(program.lineage.parents)

        if not parent_ids:
            logger.info(
                "[LineageStage:SharedBenchmark] program={} n_parents=0 "
                "(skipped: no parents)",
                program.id[:8],
            )
            return ProgramStageResult.skipped(
                message="no parents", stage=self.stage_name
            )

        child_by_g = await self._tracker.metrics_by_d(program.id)
        child_faced = set(child_by_g.keys())

        kept_ids: list[str] = []
        evidence: list[TransitionEvidence] = []

        for pid in parent_ids:
            parent_by_g = await self._tracker.metrics_by_d(pid)
            shared = child_faced & set(parent_by_g.keys())
            if len(shared) < self._min_shared:
                continue

            kept_ids.append(pid)
            if self._inject_shared_evidence:
                sp, sc, counts = _aggregate_shared_metrics(
                    parent_by_g, child_by_g, shared, self._metrics_context
                )
                evidence.append(
                    TransitionEvidence(
                        parent_id=pid,
                        shared_opponent_ids=sorted(shared),
                        shared_parent_metrics=sp,
                        shared_child_metrics=sc,
                        per_metric_shared_count=counts,
                    )
                )

        logger.info(
            "[LineageStage:SharedBenchmark] program={} kept {}/{} parents (min_shared={})",
            program.id[:8],
            len(kept_ids),
            len(parent_ids),
            self._min_shared,
        )

        if not kept_ids:
            return ProgramStageResult.skipped(
                message="no parents share eval benchmark", stage=self.stage_name
            )

        return {
            "parents": await self.storage.mget(kept_ids),
            "evidence": evidence if self._inject_shared_evidence else None,
        }
