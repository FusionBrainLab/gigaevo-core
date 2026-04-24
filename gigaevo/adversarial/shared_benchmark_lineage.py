"""Shared-benchmark filter for D-side LineageStage.

Rationale
---------
D's LineageStage narrates parent-D → child-D fitness transitions to the
mutation LLM. In adversarial co-evolution with a rotating G Hall-of-Fame
the raw whole-eval deltas are apples-to-oranges: parent and child are
usually scored against different G's. This stage subclasses LineageStage
and overrides ``preprocess()`` to:

  1) Filter out parents that share fewer than ``min_shared`` evaluation
     opponents with the child (``tracker.metrics_by_d`` key intersection).
  2) Re-run the population's aggregation logic on the per-opponent records
     restricted to the shared-G subset, via an injected
     :class:`MetricsAggregator`. The aggregator returns a dict whose schema
     matches ``program.metrics`` — so ``MetricsFormatter`` can render it
     into the prompt without a ``KeyError``.

Aggregator DI contract
----------------------
The aggregator is required (``aggregator=None`` ⇒ ``ValueError``). It is
the single validity gate: every per-opponent record in the shared-G subset
is forwarded to it, invalid entries and all; the aggregator's
``metrics_context.is_valid`` decides what to keep.

``per_metric_shared_count`` uses ``len(shared_opponent_ids)`` as a uniform
denominator for every aggregator output key. The per-metric filtering
that used to live here is now the aggregator's concern — the stage stops
leaking that detail to ``TransitionEvidence``.

Installation
------------
:class:`AdversarialAsymmetricPipelineBuilder` installs this on D runs only
via ``replace_stage("LineageStage", ...)``. Node name stays
``"LineageStage"`` so downstream edges are preserved.

Future work
-----------
G-side analog deferred — see ``experiments/IDEAS.yaml``:
``heilbron-g-side-lineage-filter``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.evolution.engine.snapshot import get_current_snapshot
from gigaevo.llm.agents.lineage import TransitionEvidence
from gigaevo.programs.core_types import ProgramStageResult, StageIO
from gigaevo.programs.metrics.aggregators import MetricsAggregator
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.stages.insights_lineage import LineageStage

if TYPE_CHECKING:
    from gigaevo.adversarial.dg_tracker import DGImprovementTracker


class SharedBenchmarkFilteredLineageStage(LineageStage):
    """LineageStage variant: filter parents by shared eval benchmark.

    A parent survives iff
    ``|tracker.faced_by_d(child) ∩ tracker.faced_by_d(parent)| >= min_shared``.
    For survivors, calls ``aggregator.aggregate`` twice (parent intrinsic,
    child intrinsic) on the shared-G subset of per-opponent records, and
    packs the results into :class:`TransitionEvidence`.

    Cache invariant
    ---------------
    ``compute_hash`` suffixes the base hash with the current engine
    ``refresh_pass``, read from the :class:`EngineSnapshot` mirror in
    ``gigaevo.evolution.engine.snapshot``. The steady-state engine bumps
    ``refresh_pass`` via ``_write_snapshot`` before each archive-refresh
    pass (see ``SteadyStateEngineConfig.refresh_passes``). Pass-2
    re-evaluations cache-miss relative to pass 1 — this closes the
    two-sided cross-program tracker race: pass 1 re-runs
    ``DGTrackerStage``, pass 2 re-runs this stage against the
    globally-fresh tracker. Within a single pass the counter is constant,
    so normal input-hash caching still deduplicates work across
    concurrently-refreshing siblings. See
    ``docs/superpowers/specs/2026-04-24-engine-snapshot-design.md``.
    """

    @classmethod
    def compute_hash(cls, params: StageIO) -> str | None:
        """Suffix the base hash with the current engine ``refresh_pass``.

        ``refresh_pass`` is the generic engine counter maintained by
        :class:`EvolutionEngine` via ``_write_snapshot``; the snapshot is
        mirrored into process memory so this sync classmethod can read it
        without awaiting.
        """
        base = super().compute_hash(params)
        if base is None:
            return None
        rp = get_current_snapshot().refresh_pass
        return f"{base}:rp{rp}"

    def __init__(
        self,
        *,
        tracker: DGImprovementTracker,
        aggregator: MetricsAggregator,
        metrics_context: MetricsContext,
        min_shared: int = 1,
        inject_shared_evidence: bool = True,
        **kwargs: Any,
    ):
        if aggregator is None:
            raise ValueError(
                "SharedBenchmarkFilteredLineageStage.aggregator is required — "
                "no silent fallback. Wire it through config/aggregator/*.yaml."
            )
        # min_shared < 1 would keep parents whose shared-opponent set is
        # empty and then hand the LLM an evidence block backed by zero
        # records. Reject it loudly rather than silently degrading.
        if min_shared < 1:
            raise ValueError(
                f"min_shared must be >= 1 (got {min_shared}); use the base "
                "LineageStage for unfiltered lineage narratives."
            )
        super().__init__(metrics_context=metrics_context, **kwargs)
        self._tracker = tracker
        self._aggregator = aggregator
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

        # Decide parent survival against the shared-opponent rule before we
        # pay the storage.mget cost — that keeps the data fetch aligned
        # with the set of parents we'll actually hand to the aggregator.
        kept: list[tuple[str, set[str]]] = []
        for pid in parent_ids:
            parent_by_g = await self._tracker.metrics_by_d(pid)
            shared = child_faced & set(parent_by_g.keys())
            if len(shared) < self._min_shared:
                continue
            kept.append((pid, shared))

        logger.info(
            "[LineageStage:SharedBenchmark] program={} kept {}/{} parents (min_shared={})",
            program.id[:8],
            len(kept),
            len(parent_ids),
            self._min_shared,
        )

        if not kept:
            return ProgramStageResult.skipped(
                message="no parents share eval benchmark", stage=self.stage_name
            )

        kept_ids = [pid for pid, _ in kept]
        parents = await self.storage.mget(kept_ids)
        parents_by_id = {p.id: p for p in parents}

        evidence: list[TransitionEvidence] | None
        if not self._inject_shared_evidence:
            evidence = None
        else:
            evidence = []
            output_keys = self._aggregator.output_keys
            for pid, shared in kept:
                parent_prog = parents_by_id.get(pid)
                if parent_prog is None:
                    # Parent was filtered out by storage (e.g. missing).
                    # Skip silently — the pipeline will still narrate the
                    # other kept parents and downstream stages tolerate an
                    # empty evidence list.
                    continue
                shared_sorted = sorted(shared)
                # Refetch tracker data per-parent so we have the shared-G
                # subset fresh and paired with the parent's hash.
                parent_by_g = await self._tracker.metrics_by_d(pid)
                parent_records = [parent_by_g[g] for g in shared_sorted]
                child_records = [child_by_g[g] for g in shared_sorted]

                parent_output = self._aggregator.aggregate(
                    parent_records, parent_prog.metrics
                )
                child_output = self._aggregator.aggregate(
                    child_records, program.metrics
                )
                evidence.append(
                    TransitionEvidence(
                        parent_id=pid,
                        shared_opponent_ids=shared_sorted,
                        shared_parent_metrics=parent_output,
                        shared_child_metrics=child_output,
                        per_metric_shared_count={
                            k: len(shared_sorted) for k in output_keys
                        },
                    )
                )

        return {
            "parents": parents,
            "evidence": evidence,
        }
