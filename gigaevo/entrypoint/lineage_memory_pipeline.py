"""Pipeline builders for guided mutation and memory-guided mutation.

Two related builders share most of their DAG structure:

* :class:`GuidedMutationPipelineBuilder` (used by ``pipeline=guided``) is the
  base. It runs a per-parent history summary via ``IntraMemoryStage`` and
  prescriptive ``MutationSuggestionStage``, with NO cross-run memory-card
  retrieval:

    - The card and the suggester live inside the DAG.
    - The history summary's ``StringContainer`` output wires DIRECTLY into
      ``MutationContextStage.memory``; there is no ``ConcatMemoryStage`` (no
      second channel to join) and no ``MemoryContextStage`` (the source of
      cross-run memory cards is dropped entirely).
    - No read/write memory hook is wired by the matching pipeline YAML. Memory
      write cadence lives under ``memory/write``.

* :class:`MemoryGuidedMutationPipelineBuilder` (used by
  ``pipeline=memory_guided``) is a subclass that re-adds memory-card
  retrieval: re-introduces ``MemoryContextStage`` and feeds the selected
  cross-run cards ONLY to ``MutationSuggestionStage`` — the suggester is the
  single source of hints for the mutator; cards never reach the mutation prompt
  verbatim. Live memory writes are still configured by ``memory/write=live``,
  not by this pipeline.

DAG-native layout (shared):

* ``DescendantProgramIds`` (collector, configured with
  ``max_selected = intra_max_children``) collects the
  current program X's already-evaluated children and emits their ids as a
  ``StringList`` keyed by the upstream-hash framework cache. When the
  engine's ``ParentRefresher`` flips X DONE→QUEUED after a new child finishes
  evaluating, the selector returns a new id list, which propagates as a
  cache-invalidating input change to ``IntraMemoryStage``.

* ``IntraMemoryStage`` (strong LLM, **descriptive only**) consumes
  ``DescendantProgramIds``'s ids and renders a per-parent lineage card
  summarising "what was tried, how each cluster fared (including HOW
  failures failed)" into ``X.metadata['intra_memory_card']``. The framework's
  ``InputHashCache`` skips the LLM whenever the children id list is
  unchanged. The card carries NO forward-looking hints — those belong to the
  next stage.

* ``MutationSuggestionStage`` (strong LLM, **prescriptive**) consumes the
  intra card (+ memory cards in the extra variant + an ancestral-momentum
  trail it walks from storage internally) and emits structured
  ``ProgramInsights`` into ``MutationContextStage``'s ``insights`` slot —
  wire-compatible with the legacy ``InsightsStage`` so the mutator prompt's
  PROGRAM INSIGHTS section is unchanged.

Both strong-LLM stages are gated on validator success (and, when the
archive gate is enabled, also on archive acceptance) so paid LLM tokens
are never spent on programs that won't enter the archive — neither card
nor suggestions are ever consumed for rejected programs.

The external-memory read half is purely a DAG feature:
``MemoryContextStage`` calls the configured ``memory.provider``. The write half
is an engine-hook feature configured separately by ``memory/write``.
"""

from __future__ import annotations

from loguru import logger

from gigaevo.entrypoint.constants import (
    DEFAULT_OPTIMIZATION_TIME_BUDGET_FRACTION,
    DEFAULT_SIMPLE_STAGE_TIMEOUT,
    MAX_CODE_LENGTH,
)
from gigaevo.entrypoint.default_pipelines import (
    DefaultPipelineBuilder,
    PipelineBuilder,
    PipelineFeature,
)
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.memory.provider import NullMemoryProvider
from gigaevo.programs.dag.automata import ExecutionOrderDependency
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.stages.ancestry_selector import AncestrySelector
from gigaevo.programs.stages.collector import DescendantProgramIds
from gigaevo.programs.stages.lineage_memory import IntraMemoryStage
from gigaevo.programs.stages.memory_context import (
    MemoryContextStage,
    MemoryExposureCounter,
)
from gigaevo.programs.stages.mutation_context import MutationContextStage
from gigaevo.programs.stages.mutation_suggestions import MutationSuggestionStage

DEFAULT_INTRA_MAX_CHILDREN = 24


class GuidedMutationFeedbackFeature(PipelineFeature):
    """Per-parent history summary plus prescriptive mutation suggestions.

    This feature owns the guided-feedback DAG slice used by both
    ``pipeline=guided`` and ``pipeline=memory_guided``: collect recent children,
    summarize them with ``IntraMemoryStage``, turn that descriptive card into
    structured suggestions, and feed the mutator through the standard
    ``MutationContextStage`` slots.
    """

    name = "guided_mutation_feedback"
    description = (
        "Build an intra-run parent-history card and convert it into structured "
        "mutation suggestions."
    )

    def __init__(
        self,
        *,
        max_insights: int,
        intra_max_children: int,
        mutation_mode: str | None,
        memory_block_last: bool,
    ):
        self._max_insights = max_insights
        self._intra_max_children = intra_max_children
        self._mutation_mode = mutation_mode
        self._memory_block_last = memory_block_last

    def apply(self, builder: PipelineBuilder) -> None:
        metrics_context = builder.ctx.problem_ctx.metrics_context
        storage = builder.ctx.storage
        strong_llm = builder.ctx.llm_wrapper
        stage_timeout = builder._stage_timeout
        task_description = builder.ctx.problem_ctx.task_description

        # Add a descendant selector tailored to intra-memory analysis: a
        # chronological recency window, NOT best_fitness — ranking by fitness
        # would drop exactly the failed and regressed children whose failure
        # modes the card exists to report, and would lose chronology.
        intra_descendant_selector = AncestrySelector(
            metrics_context=metrics_context,
            strategy="recent",
            max_selected=self._intra_max_children,
        )
        builder.replace_stage(
            "DescendantProgramIds",
            lambda: DescendantProgramIds(
                storage=storage,
                selector=intra_descendant_selector,
                timeout=stage_timeout,
            ),
        )

        # memory_block_last moves the memory block to the composite context's
        # end, adjacent to the trailing mutation instruction.
        if self._memory_block_last:
            builder.replace_stage(
                "MutationContextStage",
                lambda: MutationContextStage(
                    metrics_context=metrics_context,
                    timeout=stage_timeout,
                    memory_last=True,
                ),
            )

        builder.add_stage(
            "IntraMemoryStage",
            lambda: IntraMemoryStage(
                llm=strong_llm,
                storage=storage,
                metrics_context=metrics_context,
                max_children=self._intra_max_children,
                task_description=task_description,
                timeout=stage_timeout,
            ),
        )
        builder.add_stage(
            "MutationSuggestionStage",
            lambda: MutationSuggestionStage(
                llm=strong_llm,
                storage=storage,
                metrics_context=metrics_context,
                task_description=task_description,
                max_insights=self._max_insights,
                timeout=stage_timeout,
                mutation_mode=self._mutation_mode,
            ),
        )

        # Rewire MutationContextStage with the intra-only descriptive/prescriptive split:
        #   * IntraMemoryStage is DESCRIPTIVE and consumes only ``children_ids``.
        #   * MutationSuggestionStage is PRESCRIPTIVE: it takes the intra card
        #     (+ ancestral trail walked from storage internally) and emits
        #     structured ``ProgramInsights`` into MutationContextStage's
        #     ``insights`` slot — same shape the legacy ``InsightsStage``
        #     produced, so the mutator's PROGRAM INSIGHTS section renders
        #     unchanged.
        #   * The intra card wires DIRECTLY into MutationContextStage.memory
        #     (Box[str] -> StringContainer | None) — no ConcatMemoryStage
        #     needed when there's no second channel to join.
        builder.add_data_flow_edge(
            "DescendantProgramIds", "IntraMemoryStage", "children_ids"
        )
        builder.add_data_flow_edge(
            "IntraMemoryStage", "MutationSuggestionStage", "intra_card"
        )
        builder.add_data_flow_edge(
            "EvolutionaryStatisticsCollector",
            "MutationSuggestionStage",
            "evolutionary_statistics",
        )
        builder.add_data_flow_edge(
            "MutationSuggestionStage", "MutationContextStage", "insights"
        )
        builder.add_data_flow_edge("IntraMemoryStage", "MutationContextStage", "memory")

        builder.add_exec_dep(
            "IntraMemoryStage",
            ExecutionOrderDependency.on_success("CallValidatorFunction"),
        )
        builder.add_exec_dep(
            "IntraMemoryStage",
            ExecutionOrderDependency.always_after("EnsureMetricsStage"),
        )
        builder.add_exec_dep(
            "IntraMemoryStage",
            ExecutionOrderDependency.always_after("DescendantProgramIds"),
        )
        builder.add_exec_dep(
            "MutationSuggestionStage",
            ExecutionOrderDependency.on_success("CallValidatorFunction"),
        )
        builder.add_exec_dep(
            "MutationSuggestionStage",
            ExecutionOrderDependency.always_after("EnsureMetricsStage"),
        )
        builder.add_exec_dep(
            "MutationSuggestionStage",
            ExecutionOrderDependency.always_after("IntraMemoryStage"),
        )
        builder.add_exec_dep(
            "MutationSuggestionStage",
            ExecutionOrderDependency.always_after("EvolutionaryStatisticsCollector"),
        )

        if builder._archive_gate_enabled:
            builder.add_exec_dep(
                "IntraMemoryStage",
                ExecutionOrderDependency.on_success("ArchivePotentialGateStage"),
            )
            builder.add_exec_dep(
                "MutationSuggestionStage",
                ExecutionOrderDependency.on_success("ArchivePotentialGateStage"),
            )


class ExternalMemoryCardReadFeature(PipelineFeature):
    """Select external memory cards and feed them only to the suggester.

    This feature is the read half of the memory system. It installs
    ``MemoryContextStage`` and wires selected cards to
    ``MutationSuggestionStage.memory_cards``. It deliberately does not touch
    writer hooks; write cadence belongs to ``memory/write``.
    """

    name = "external_memory_card_read"
    description = (
        "Retrieve cross-run memory cards through memory.provider and expose "
        "them to MutationSuggestionStage only."
    )

    def __init__(
        self,
        *,
        fresh_context_reorder: bool,
        reverse_repack: bool,
        no_card_control_probability: float,
    ):
        self._fresh_context_reorder = fresh_context_reorder
        self._reverse_repack = reverse_repack
        self._no_card_control_probability = no_card_control_probability

    def apply(self, builder: PipelineBuilder) -> None:
        memory_provider = builder.ctx.memory_provider
        if isinstance(memory_provider, NullMemoryProvider):
            logger.warning(
                "[Memory][Arm] read path DISABLED (memory=none) under "
                "pipeline=memory_guided — selected memory cards cannot flow"
            )

        task_description = builder.ctx.problem_ctx.task_description
        metrics_context = builder.ctx.problem_ctx.metrics_context
        metrics_description = MetricsFormatter(
            metrics_context
        ).format_metrics_description()
        stage_timeout = builder._stage_timeout

        # One exposure counter is shared across all per-program stage instances.
        exposure = MemoryExposureCounter()
        builder.add_stage(
            "MemoryContextStage",
            lambda: MemoryContextStage(
                memory_provider=memory_provider,
                task_description=task_description,
                metrics_description=metrics_description,
                metrics_context=metrics_context,
                timeout=stage_timeout,
                exposure=exposure,
                fresh_context_reorder=self._fresh_context_reorder,
                reverse_repack=self._reverse_repack,
                no_card_control_probability=self._no_card_control_probability,
            ),
        )

        builder.add_data_flow_edge(
            "MemoryContextStage", "MutationSuggestionStage", "memory_cards"
        )
        builder.add_exec_dep(
            "MutationSuggestionStage",
            ExecutionOrderDependency.always_after("MemoryContextStage"),
        )
        builder.add_exec_dep(
            "MemoryContextStage",
            ExecutionOrderDependency.on_success("CallValidatorFunction"),
        )
        if builder._archive_gate_enabled:
            builder.add_exec_dep(
                "MemoryContextStage",
                ExecutionOrderDependency.on_success("ArchivePotentialGateStage"),
            )

        # GAM-fresh-context reorder (Arm C). Condition external-memory card
        # selection on the SAME fresh this-pass signals the mutation agent sees
        # — the lineage card + live evolutionary snapshot — instead of a
        # one-pass-stale metadata block. Gated so Arm B
        # (fresh_context_reorder=False) reverts to the pre-reorder DAG and the
        # stage passes parent_context=None (metadata fallback).
        if self._fresh_context_reorder:
            builder.add_data_flow_edge(
                "IntraMemoryStage", "MemoryContextStage", "intra_card"
            )
            builder.add_data_flow_edge(
                "EvolutionaryStatisticsCollector",
                "MemoryContextStage",
                "evolutionary_statistics",
            )
            builder.add_exec_dep(
                "MemoryContextStage",
                ExecutionOrderDependency.always_after("IntraMemoryStage"),
            )
            builder.add_exec_dep(
                "MemoryContextStage",
                ExecutionOrderDependency.always_after(
                    "EvolutionaryStatisticsCollector"
                ),
            )


class GuidedMutationPipelineBuilder(DefaultPipelineBuilder):
    """Default pipeline + DAG-native guided mutation feedback.

    Inherits from :class:`DefaultPipelineBuilder`; contextual problems get
    ``AddContext`` automatically from the base builder, while non-contextual
    problems keep the lean source-evaluation path.

    Used by ``pipeline=guided``. External-memory writing is independent of this
    builder; external-memory reading requires ``pipeline=memory_guided``.
    """

    _reads_memory_cards = False

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 3600.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
        max_parallel: int | None = None,
        max_insights: int = 5,
        max_code_length: int = MAX_CODE_LENGTH,
        archive_gate_enabled: bool = False,
        intra_max_children: int = DEFAULT_INTRA_MAX_CHILDREN,
        mutation_mode: str | None = None,
        enable_optuna_stage: bool = False,
        optimization_time_budget: float | None = None,
        memory_block_last: bool = False,
        program_format_feature: PipelineFeature | None = None,
    ):
        super().__init__(
            ctx,
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
            max_parallel=max_parallel,
            max_insights=max_insights,
            max_code_length=max_code_length,
            archive_gate_enabled=archive_gate_enabled,
            include_legacy_feedback=False,
            program_format_feature=program_format_feature,
        )
        self._enable_optuna_stage = enable_optuna_stage
        self._optimization_time_budget_arg = optimization_time_budget
        self._dag_timeout_arg = dag_timeout
        self.apply_feature(
            GuidedMutationFeedbackFeature(
                max_insights=max_insights,
                intra_max_children=intra_max_children,
                mutation_mode=mutation_mode,
                memory_block_last=memory_block_last,
            )
        )

        if self._enable_optuna_stage:
            self._optimization_time_budget = (
                self._optimization_time_budget_arg
                if self._optimization_time_budget_arg is not None
                else self._dag_timeout_arg * DEFAULT_OPTIMIZATION_TIME_BUDGET_FRACTION
            )
            self._wire_optuna_stage()

        # The subclass sets _reads_memory_cards=True, so this never fires
        # during its construction even though it runs this __init__.
        if not self._reads_memory_cards and not isinstance(
            ctx.memory_provider, NullMemoryProvider
        ):
            logger.warning(
                "[Memory][Arm] memory provider {} is configured but this "
                "pipeline never reads cards — selected cards reach no stage; "
                "use pipeline=memory_guided to consume them",
                type(ctx.memory_provider).__name__,
            )


class MemoryGuidedMutationPipelineBuilder(GuidedMutationPipelineBuilder):
    """Guided mutation base + external memory-card retrieval.

    Used by ``pipeline=memory_guided``. Re-adds the
    :class:`MemoryContextStage` that the intra-only base strips and wires its
    selected cards into ``MutationSuggestionStage.memory_cards`` ONLY — the
    suggester digests them into structured ``ProgramInsights``; the mutator
    never sees card text verbatim. ``MutationContextStage.memory`` keeps the
    bare per-parent history summary via the base class's direct edge, identical
    to ``pipeline=guided``. This class does not wire writer hooks; use
    ``memory/write=live`` for live writes.

    Common pairings:

        memory=reader           — read a pre-built bank; no writer.
        memory=full             — read + end-of-run writer by default.
        memory=full memory/write=live
                                — read + live writer refresh.

    Verify ``.hydra/config.yaml`` does not show ``NullMemoryProvider`` before
    trusting memory-guided results.
    """

    _reads_memory_cards = True

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 3600.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
        max_parallel: int | None = None,
        max_insights: int = 5,
        max_code_length: int = MAX_CODE_LENGTH,
        archive_gate_enabled: bool = False,
        intra_max_children: int = DEFAULT_INTRA_MAX_CHILDREN,
        mutation_mode: str | None = None,
        enable_optuna_stage: bool = False,
        optimization_time_budget: float | None = None,
        fresh_context_reorder: bool = True,
        reverse_repack: bool = False,
        no_card_control_probability: float = 0.0,
        memory_block_last: bool = False,
        program_format_feature: PipelineFeature | None = None,
    ):
        super().__init__(
            ctx,
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
            max_parallel=max_parallel,
            max_insights=max_insights,
            max_code_length=max_code_length,
            archive_gate_enabled=archive_gate_enabled,
            intra_max_children=intra_max_children,
            mutation_mode=mutation_mode,
            enable_optuna_stage=enable_optuna_stage,
            optimization_time_budget=optimization_time_budget,
            memory_block_last=memory_block_last,
            program_format_feature=program_format_feature,
        )

        self.apply_feature(
            ExternalMemoryCardReadFeature(
                fresh_context_reorder=fresh_context_reorder,
                reverse_repack=reverse_repack,
                no_card_control_probability=no_card_control_probability,
            )
        )
