"""Adversarial asymmetric pipeline builder.

Extends AdversarialPipelineBuilder:
  - D runs: adds SourceCodeInjectionStage (forked from FetchOpponentIdsStage)
  - G runs, Arm C: adds GradientInPromptStage (reads D's archive)
  - G runs, Arm A: composition injection handled as pre-step hook (not pipeline)

Key data flow (D runs):
  FetchOpponentIdsStage → FetchOpponentResultsStage (evaluation)
  FetchOpponentIdsStage → SourceCodeInjectionStage (mutation prompt)

Parametric: n_opponents=k, source_prompt_k=l (l<=k).
"""

from __future__ import annotations

from typing import Literal

from loguru import logger

from gigaevo.adversarial.dg_tracker_stage import DGTrackerStage
from gigaevo.adversarial.gradient_prompt import GradientInPromptStage
from gigaevo.adversarial.opponent_provider import OpponentArchiveProvider
from gigaevo.adversarial.pipeline import AdversarialPipelineBuilder
from gigaevo.adversarial.shared_benchmark_lineage import (
    DGTrackerSharedOpponentResolver,
    SharedBenchmarkLineageStage,
)
from gigaevo.adversarial.source_injection import SourceCodeInjectionStage
from gigaevo.adversarial.tracker_coverage_stages import (
    ComputeDWinsCountStage,
    ComputeGResistedCountStage,
)
from gigaevo.entrypoint.constants import DEFAULT_SIMPLE_STAGE_TIMEOUT
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.programs.dag.automata import ExecutionOrderDependency
from gigaevo.programs.stages.json_processing import MergeDictStage


class AdversarialAsymmetricPipelineBuilder(AdversarialPipelineBuilder):
    """Asymmetric adversarial pipeline with source injection and gradient feedback.

    For D (Improver) runs:
      - SourceCodeInjectionStage receives opponent IDs from FetchOpponentIdsStage
        (same IDs used for evaluation) and shows top-l source codes in D's
        mutation prompt.

    For G (Constructor) runs with feedback_mode="gradient_in_prompt" (Arm C):
      - GradientInPromptStage reads D's best from D's archive and shows it
        in G's mutation prompt.

    For G (Constructor) runs with feedback_mode="composition" (Arm A):
      - No pipeline-level changes. CompositionInjectionHook is wired at the
        engine level (not here).

    Args:
        ctx: Evolution context.
        opponent_provider: Archive provider for the opponent population.
        d_provider: Archive provider for D's population (needed for Arm C G runs).
        population_role: "constructor" or "improver".
        feedback_mode: "composition" (Arm A) or "gradient_in_prompt" (Arm C).
        n_opponents: Number of opponents for evaluation (k).
        source_prompt_k: Number of source codes to show in D's prompt (l <= k).
        per_opponent_timeout: Timeout per opponent execution.
        fallback_dir: Directory with fallback opponent codes.
        archive_reeval: Whether to use InputHashCache for opponent results.
        dag_timeout: Total DAG execution timeout.
        stage_timeout: Per-stage execution timeout.
    """

    def __init__(
        self,
        ctx: EvolutionContext,
        opponent_provider: OpponentArchiveProvider,
        d_provider: OpponentArchiveProvider | None = None,
        population_role: Literal["constructor", "improver"] = "constructor",
        feedback_mode: Literal["composition", "gradient_in_prompt"] = "composition",
        n_opponents: int = 1,
        source_prompt_k: int = 1,
        per_opponent_timeout: float = 10.0,
        fallback_dir: str = "fallback",
        archive_reeval: bool = False,
        dg_tracker: object = None,
        *,
        dag_timeout: float = 7200.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
    ):
        super().__init__(
            ctx,
            opponent_provider,
            n_opponents,
            per_opponent_timeout,
            fallback_dir,
            archive_reeval,
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
        )

        if population_role == "improver":
            self._add_source_injection(
                opponent_provider, source_prompt_k, stage_timeout
            )

        if population_role == "constructor" and feedback_mode == "gradient_in_prompt":
            if d_provider is None:
                raise ValueError(
                    "gradient_in_prompt feedback mode requires d_provider "
                    "(OpponentArchiveProvider pointing to D's archive)"
                )
            self._add_gradient_prompt(d_provider, dg_tracker, stage_timeout)

        # Wire DGTrackerStage to record per-opponent fitness deltas into tracker.
        if dg_tracker is not None:
            self._add_dg_tracker_stage(dg_tracker, population_role, stage_timeout)
            self._add_tracker_coverage_stages(
                dg_tracker, population_role, stage_timeout
            )
            if population_role == "improver":
                self._add_shared_benchmark_lineage(ctx, dg_tracker, stage_timeout)

        # Wire cache_on edges so InsightsStage/LineageStage invalidate when the
        # opponent set rotates. The InputsModel of each is CacheOnlyInput, so
        # the field value (opponent IDs) is folded into the cache hash without
        # affecting compute(). Guarded so non-default pipelines stay safe.
        self._wire_cache_on_edges()

        logger.info(
            "[AsymmetricPipeline] role={} feedback={} n_opp={} source_prompt_k={} "
            "dg_tracker={}",
            population_role,
            feedback_mode,
            n_opponents,
            source_prompt_k,
            "yes" if dg_tracker is not None else "no",
        )

    def _wire_cache_on_edges(self) -> None:
        """Attach FetchOpponentIdsStage output as cache_on for LLM stages.

        Idempotent and safe: only fires when the target stage is registered.
        """
        if "InsightsStage" in self._nodes:
            self.add_data_flow_edge(
                "FetchOpponentIdsStage", "InsightsStage", "cache_on"
            )
        if "LineageStage" in self._nodes:
            self.add_data_flow_edge("FetchOpponentIdsStage", "LineageStage", "cache_on")
        if "SharedBenchmarkLineageStage" in self._nodes:
            self.add_data_flow_edge(
                "FetchOpponentIdsStage", "SharedBenchmarkLineageStage", "cache_on"
            )

    def _add_source_injection(
        self,
        opponent_provider: OpponentArchiveProvider,
        source_prompt_k: int,
        stage_timeout: float,
    ) -> None:
        self.remove_data_flow_edge("FormatterStage", "MutationContextStage")
        self.add_stage(
            "SourceCodeInjectionStage",
            lambda: SourceCodeInjectionStage(
                opponent_provider=opponent_provider,
                source_prompt_k=source_prompt_k,
                timeout=stage_timeout,
            ),
        )
        self.add_data_flow_edge(
            "FetchOpponentIdsStage", "SourceCodeInjectionStage", "opponent_ids"
        )
        self.add_data_flow_edge(
            "SourceCodeInjectionStage", "MutationContextStage", "formatted"
        )
        self.add_exec_dep(
            "SourceCodeInjectionStage",
            ExecutionOrderDependency.on_success("FetchOpponentIdsStage"),
        )

    def _add_gradient_prompt(
        self,
        d_provider: OpponentArchiveProvider,
        dg_tracker: object | None,
        stage_timeout: float,
    ) -> None:
        provider = d_provider  # Capture for lambda closure.
        tracker = dg_tracker  # Capture for lambda closure.

        self.remove_data_flow_edge("FormatterStage", "MutationContextStage")
        self.add_stage(
            "GradientInPromptStage",
            lambda: GradientInPromptStage(
                opponent_provider=provider,
                dg_tracker=tracker,
                timeout=stage_timeout,
            ),
        )
        self.add_data_flow_edge(
            "GradientInPromptStage", "MutationContextStage", "formatted"
        )
        self.add_exec_dep(
            "GradientInPromptStage",
            ExecutionOrderDependency.on_success("ValidateCodeStage"),
        )

    def _add_dg_tracker_stage(
        self, dg_tracker: object, population_role: str, stage_timeout: float
    ) -> None:
        """Add DGTrackerStage to record per-opponent fitness deltas into the tracker.

        This stage is a pass-through that extracts per-opponent fitness deltas from
        the validator artifact and records them into the DGImprovementTracker for
        feedback pathways (gradient_in_prompt, composition injection).

        Args:
            dg_tracker: DGImprovementTracker instance.
            population_role: "constructor" or "improver".
            stage_timeout: Per-stage execution timeout.
        """
        tracker = dg_tracker  # Capture for lambda closure.
        role = population_role  # Capture for lambda closure.
        timeout = stage_timeout  # Capture for lambda closure.

        def make_stage():
            return DGTrackerStage(
                dg_tracker=tracker,
                role=role,
                timeout=timeout,
            )

        self.add_stage("DGTrackerStage", make_stage)
        # Wire: opponent_ids from FetchOpponentIdsStage, validation_result from CallValidatorFunction
        self.add_data_flow_edge(
            "FetchOpponentIdsStage", "DGTrackerStage", "opponent_ids"
        )
        self.add_data_flow_edge(
            "CallValidatorFunction", "DGTrackerStage", "validation_result"
        )
        # Execution: run after CallValidatorFunction completes
        self.add_exec_dep(
            "DGTrackerStage",
            ExecutionOrderDependency.on_success("CallValidatorFunction"),
        )

    def _add_tracker_coverage_stages(
        self,
        dg_tracker: object,
        population_role: Literal["constructor", "improver"],
        stage_timeout: float,
    ) -> None:
        """Add tracker coverage stages + route their output into the candidate dict.

        The default pipeline wires ``MergeMetricsStage → EnsureMetricsStage`` via
        a data_flow_edge named ``candidate``. ``EnsureMetricsStage`` validates
        that dict against :class:`MetricsContext` required keys — including
        ``wins`` for v3 BD axes.  ``wins`` is produced by the coverage stage,
        which runs in parallel with ``MergeMetricsStage`` and would be invisible
        to ``EnsureMetricsStage`` if we only wrote to ``program.metrics``.

        Fix: insert a ``MergeCoverageMetricsStage`` (a :class:`MergeDictStage`)
        between ``MergeMetricsStage`` and ``EnsureMetricsStage`` and route the
        coverage stage's ``FloatDictContainer`` output into it as ``second``
        (second overrides first on key collision). Data-flow edges give us the
        exec-order constraint for free.
        """
        coverage_stage_name = (
            "ComputeDWinsCountStage"
            if population_role == "improver"
            else "ComputeGResistedCountStage"
        )

        if population_role == "improver":

            def make_coverage_stage():
                return ComputeDWinsCountStage(
                    dg_tracker=dg_tracker, timeout=stage_timeout
                )
        else:

            def make_coverage_stage():
                return ComputeGResistedCountStage(
                    dg_tracker=dg_tracker, timeout=stage_timeout
                )

        self.add_stage(coverage_stage_name, make_coverage_stage)
        self.add_exec_dep(
            coverage_stage_name,
            ExecutionOrderDependency.on_success("DGTrackerStage"),
        )

        # Rewire MergeMetricsStage → MergeCoverageMetricsStage → EnsureMetricsStage.
        # This guarantees `wins` is in the candidate dict EnsureMetricsStage reads,
        # fixing `ValueError: Missing required metric keys: ['wins']`.
        _coverage_timeout = stage_timeout

        def make_coverage_merge():
            return MergeDictStage[str, float](timeout=_coverage_timeout)

        self.add_stage("MergeCoverageMetricsStage", make_coverage_merge)
        self.remove_data_flow_edge("MergeMetricsStage", "EnsureMetricsStage")
        self.add_data_flow_edge(
            "MergeMetricsStage", "MergeCoverageMetricsStage", "first"
        )
        self.add_data_flow_edge(
            coverage_stage_name, "MergeCoverageMetricsStage", "second"
        )
        self.add_data_flow_edge(
            "MergeCoverageMetricsStage", "EnsureMetricsStage", "candidate"
        )

    def _add_shared_benchmark_lineage(
        self,
        ctx: EvolutionContext,
        dg_tracker: object,
        stage_timeout: float,
    ) -> None:
        """Add SharedBenchmarkLineageStage for D runs (§3.5 Prong 2, §9.1).

        Computes an HoF-invariant lineage trend for D programs via intersection
        of G's both child-D and parent-D have faced. Invalidates when the G HoF
        rotates (``cache_on=opponent_ids``) — new tracker pairs enter the
        shared benchmark primarily at HoF transitions. Emits ``[LINEAGE_TREND]``
        canonical event for log-based verification (§13.3).
        """
        tracker = dg_tracker
        storage = ctx.storage
        timeout = stage_timeout

        def make_stage():
            return SharedBenchmarkLineageStage(
                resolver=DGTrackerSharedOpponentResolver(tracker=tracker),
                storage=storage,
                timeout=timeout,
            )

        self.add_stage("SharedBenchmarkLineageStage", make_stage)
        self.add_exec_dep(
            "SharedBenchmarkLineageStage",
            ExecutionOrderDependency.on_success("DGTrackerStage"),
        )
