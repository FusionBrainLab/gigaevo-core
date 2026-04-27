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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from gigaevo.adversarial.dg_tracker_stage import DGTrackerStage

if TYPE_CHECKING:
    from gigaevo.adversarial.dg_tracker import DGImprovementTracker
from gigaevo.adversarial.gradient_prompt import GradientInPromptStage
from gigaevo.adversarial.opponent_provider import (
    OpponentArchiveProvider,
    OpponentSamplingMode,
)
from gigaevo.adversarial.pipeline import AdversarialPipelineBuilder
from gigaevo.adversarial.shared_benchmark_lineage import (
    SharedBenchmarkFilteredLineageStage,
)
from gigaevo.adversarial.source_injection import SourceCodeInjectionStage
from gigaevo.adversarial.tracker_coverage_stages import (
    ComputeDWinsCountStage,
    ComputeGResistedCountStage,
)
from gigaevo.entrypoint.constants import DEFAULT_SIMPLE_STAGE_TIMEOUT
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.programs.dag.automata import ExecutionOrderDependency
from gigaevo.programs.metrics.aggregators import MetricsAggregator, NullAggregator
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.stages.json_processing import MergeDictStage
from gigaevo.programs.stages.python_executors.execution import ParseMetricsStage


@dataclass
class LineageFilterConfig:
    """Config for D-side SharedBenchmarkFilteredLineageStage.

    The ``aggregator`` field is REQUIRED at build time — passing
    ``LineageFilterConfig()`` (no aggregator) raises ``ValueError`` in the
    pipeline builder. To disable filtering entirely, don't install the
    filtered variant at all — use the base ``LineageStage``.
    ``inject_shared_evidence=False`` ⇒ filter still applies but no
    ``TransitionEvidence`` is emitted (ablation).

    ``aggregator`` must be an already-instantiated :class:`MetricsAggregator`.
    The pipeline builder also accepts a raw ``DictConfig`` in place of this
    dataclass; in that case the aggregator sub-config is resolved via
    ``hydra.utils.instantiate`` with ``metrics_context`` injected as kwarg.
    """

    min_shared: int = 1
    inject_shared_evidence: bool = True
    aggregator: MetricsAggregator | None = field(default=None)


def _resolve_lineage_filter(
    spec: LineageFilterConfig | DictConfig | None,
    metrics_context: MetricsContext,
) -> LineageFilterConfig:
    """Normalize the ``lineage_filter`` argument to a validated dataclass.

    Accepts either an already-built :class:`LineageFilterConfig` or a raw
    :class:`omegaconf.DictConfig`. A DictConfig must contain an
    ``aggregator`` key with a ``_target_`` (Hydra instantiates it with
    ``metrics_context`` injected as a kwarg — YAML doesn't need to
    reference the shared singleton).

    Raises ``ValueError`` if the resolved config has no aggregator.
    """
    if spec is None:
        raise ValueError(
            "lineage_filter.aggregator required — no silent fallback. "
            "Pass LineageFilterConfig(aggregator=…) or a DictConfig with an "
            "aggregator._target_."
        )

    if isinstance(spec, DictConfig):
        agg_cfg = spec.get("aggregator")
        if agg_cfg is None:
            raise ValueError(
                "lineage_filter.aggregator required — no silent fallback. "
                "Add an `aggregator:` block with a `_target_` to the pipeline "
                "config."
            )
        aggregator = hydra.utils.instantiate(agg_cfg, metrics_context=metrics_context)
        raw = OmegaConf.to_container(spec, resolve=True)
        assert isinstance(raw, dict)
        return LineageFilterConfig(
            min_shared=int(raw.get("min_shared", 1)),
            inject_shared_evidence=bool(raw.get("inject_shared_evidence", True)),
            aggregator=aggregator,
        )

    if spec.aggregator is None:
        raise ValueError("lineage_filter.aggregator required — no silent fallback.")
    return spec


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
        dg_tracker: DGImprovementTracker | None = None,
        *,
        aggregator: MetricsAggregator | None = None,
        lineage_filter: LineageFilterConfig | DictConfig | None = None,
        opponent_result_mode: Literal["exec", "cached"] = "exec",
        opponent_sampling_mode: OpponentSamplingMode | str = OpponentSamplingMode.TOP_K,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        opponent_sources: list[dict[str, int | str]] | None = None,
        dag_timeout: float = 7200.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
        cache_insights_on_opponents: bool = True,
    ):
        self._cache_insights_on_opponents = cache_insights_on_opponents
        super().__init__(
            ctx,
            opponent_provider,
            n_opponents,
            per_opponent_timeout,
            fallback_dir,
            archive_reeval,
            opponent_result_mode=opponent_result_mode,
            opponent_sampling_mode=opponent_sampling_mode,
            redis_host=redis_host,
            redis_port=redis_port,
            opponent_sources=opponent_sources,
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
                resolved_filter = _resolve_lineage_filter(
                    lineage_filter, ctx.problem_ctx.metrics_context
                )
                self._replace_lineage_with_filtered(
                    ctx,
                    dg_tracker,
                    resolved_filter,
                    stage_timeout,
                )

        # Wire cache_on edges so InsightsStage/LineageStage invalidate when the
        # opponent set rotates. The InputsModel of each is CacheOnlyInput, so
        # the field value (opponent IDs) is folded into the cache hash without
        # affecting compute(). InsightsStage edge is gated on
        # cache_insights_on_opponents (default True). See _wire_cache_on_edges.
        self._wire_cache_on_edges()

        # Default aggregator to NullAggregator when Hydra hasn't supplied one,
        # so downstream `isinstance(agg, NullAggregator)` checks work without
        # None-special-casing. The None path only exists for legacy call sites
        # that pre-date the Task 3 kwarg.
        self._aggregator: MetricsAggregator = aggregator or NullAggregator()
        if not isinstance(self._aggregator, NullAggregator):
            self._insert_parse_metrics_stage(stage_timeout)

        logger.info(
            "[AsymmetricPipeline] role={} feedback={} n_opp={} source_prompt_k={} "
            "dg_tracker={} aggregator={}",
            population_role,
            feedback_mode,
            n_opponents,
            source_prompt_k,
            "yes" if dg_tracker is not None else "no",
            type(self._aggregator).__name__,
        )

    def _insert_parse_metrics_stage(self, stage_timeout: float) -> None:
        """Insert ParseMetricsStage between CallValidatorFunction and consumers.

        evaluate.py returns `(intrinsic, artifact)`. ParseMetricsStage composes
        `program.metrics` from `artifact.per_opp_metrics` via the aggregator
        and emits the legacy `(metrics, artifact)` tuple under input_name
        `validation_result` so FetchMetrics / FetchArtifact / DGTrackerStage
        stay unchanged.

        Gated on `not isinstance(self._aggregator, NullAggregator)` — the
        null-object sentinel signals "no aggregator configured; preserve the
        legacy DAG". Non-Heilbron adversarial pipelines inherit the top-level
        `aggregator=none` default and so skip this insertion entirely.
        """
        agg = self._aggregator
        timeout = stage_timeout

        def make_parse_stage() -> ParseMetricsStage:
            return ParseMetricsStage(aggregator=agg, timeout=timeout)

        self.add_stage("ParseMetricsStage", make_parse_stage)

        # Rewrite every CallValidatorFunction→X edge so that X reads from
        # ParseMetricsStage instead.
        new_edges = []
        rewired_destinations: list[str] = []
        from gigaevo.programs.dag.automata import DataFlowEdge

        for e in self._data_flow_edges:
            if (
                e.source_stage == "CallValidatorFunction"
                and e.input_name == "validation_result"
            ):
                new_edges.append(
                    DataFlowEdge.create(
                        source="ParseMetricsStage",
                        destination=e.destination_stage,
                        input_name="validation_result",
                    )
                )
                rewired_destinations.append(e.destination_stage)
            else:
                new_edges.append(e)
        self._data_flow_edges = new_edges

        # Feed CallValidatorFunction's raw tuple into ParseMetricsStage.
        self.add_data_flow_edge(
            "CallValidatorFunction", "ParseMetricsStage", "raw_validator_output"
        )
        self.add_exec_dep(
            "ParseMetricsStage",
            ExecutionOrderDependency.on_success("CallValidatorFunction"),
        )
        # Downstream consumers now depend on ParseMetricsStage's success; keep
        # any previous "always_after CallValidatorFunction" on them — they still
        # wait for the raw call to finish, transitively via ParseMetricsStage.
        logger.info(
            "[AsymmetricPipeline] inserted ParseMetricsStage; rewired {} consumers: {}",
            len(rewired_destinations),
            sorted(set(rewired_destinations)),
        )

    def _wire_cache_on_edges(self) -> None:
        """Attach FetchOpponentIdsStage output as cache_on for LLM stages.

        Idempotent and safe: only fires when the target stage is registered.

        InsightsStage wiring is gated on `cache_insights_on_opponents` (default
        True). Disable it (`pipeline_builder.cache_insights_on_opponents=false`)
        to match v1 (PR #204) behavior, where insights were cached purely on
        program identity — opponent rotation did not invalidate them.
        Insights describe a program in isolation, so the rotation-driven
        invalidation forces an LLM re-run every generation with no semantic
        change.
        """
        if self._cache_insights_on_opponents and "InsightsStage" in self._nodes:
            self.add_data_flow_edge(
                "FetchOpponentIdsStage", "InsightsStage", "cache_on"
            )
        if "LineageStage" in self._nodes:
            self.add_data_flow_edge("FetchOpponentIdsStage", "LineageStage", "cache_on")

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
        dg_tracker: DGImprovementTracker | None,
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
        self,
        dg_tracker: DGImprovementTracker,
        population_role: str,
        stage_timeout: float,
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
        dg_tracker: DGImprovementTracker,
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

    def _replace_lineage_with_filtered(
        self,
        ctx: EvolutionContext,
        dg_tracker: DGImprovementTracker,
        cfg: LineageFilterConfig,
        stage_timeout: float,
    ) -> None:
        """Swap default LineageStage for SharedBenchmarkFilteredLineageStage on D runs.

        Node name stays ``"LineageStage"`` so all incoming/outgoing edges
        (LineagesFromAncestors / LineagesToDescendants → MutationContextStage,
        cache_on from FetchOpponentIdsStage) remain intact. The filtered
        variant subclasses LineageStage, overrides preprocess() to drop
        parents without a shared eval benchmark, and injects
        TransitionEvidence into the LLM lineage agent.
        """
        tracker = dg_tracker
        storage = ctx.storage
        llm = ctx.llm_wrapper
        task_description = ctx.problem_ctx.task_description
        metrics_context = ctx.problem_ctx.metrics_context
        prompts_dir = ctx.prompts_dir

        # cfg.aggregator has already been validated non-None by
        # _resolve_lineage_filter in the builder's __init__ — it is safe to
        # forward directly here.
        aggregator = cfg.aggregator
        assert aggregator is not None  # narrowed by _resolve_lineage_filter

        def make_stage() -> SharedBenchmarkFilteredLineageStage:
            return SharedBenchmarkFilteredLineageStage(
                llm=llm,
                task_description=task_description,
                metrics_context=metrics_context,
                storage=storage,
                prompts_dir=prompts_dir,
                tracker=tracker,
                aggregator=aggregator,
                min_shared=cfg.min_shared,
                inject_shared_evidence=cfg.inject_shared_evidence,
                timeout=stage_timeout,
            )

        self.replace_stage("LineageStage", make_stage)
        self.add_exec_dep(
            "LineageStage",
            ExecutionOrderDependency.on_success("DGTrackerStage"),
        )
