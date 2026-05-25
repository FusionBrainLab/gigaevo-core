"""Pipeline builders for the intra/extra live-memory experiment.

Two related builders share most of their DAG structure:

* :class:`IntraMemoryPipelineBuilder` (intra-only, used by ``pipeline=standard``)
  is the base. It runs a per-parent lineage card via ``IntraMemoryStage`` and
  prescriptive ``MutationSuggestionStage``, with NO cross-population memory
  channel:

    - The card and the suggester live inside the DAG.
    - The card's ``StringContainer`` output wires DIRECTLY into
      ``MutationContextStage.memory``; there is no ``ConcatMemoryStage`` (no
      second channel to join) and no ``MemoryContextStage`` (the source of
      cross-population cards is dropped entirely).
    - No ``LiveMemoryRefreshHook`` is wired by the matching YAML — when an
      end-of-run extractor is desired the user opts in with
      ``ideas_tracker=default`` (post_run_hook only; no mid-run refresh).

* :class:`IntraExtraMemoryPipelineBuilder` (used by
  ``pipeline=intra_extra_memory``) is a subclass that re-adds the extra
  channel: re-introduces ``MemoryContextStage``, adds ``ConcatMemoryStage``,
  and rewires so the joined intra-card + memory-cards block feeds
  ``MutationContextStage.memory``. The matching YAML also wires
  ``LiveMemoryRefreshHook`` so the extra cards are refreshed mid-run.

DAG-native layout (shared):

* ``DescendantProgramIds`` (collector, kept from the default pipeline but
  reconfigured with ``max_selected = intra_max_children``) collects the
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

Legacy stages stripped (superseded by intra + suggestion):

* ``AncestorProgramIds``, ``LineageStage``, ``LineagesToDescendants``,
  ``LineagesFromAncestors``, ``InsightsStage``.

The "extra" half of :class:`IntraExtraMemoryPipelineBuilder` is provided by
:class:`LiveMemoryRefreshHook` (``gigaevo/memory/live_memory_hook.py``),
which wraps :meth:`IdeaTracker.run_increment` and is wired into the engine's
``post_step_hook`` slot via ``pipeline=intra_extra_memory``. The selector
inside :class:`MemoryContextStage` surfaces the freshest cards through
reload-on-read.
"""

from __future__ import annotations

from gigaevo.entrypoint.constants import (
    DEFAULT_OPTIMIZATION_TIME_BUDGET_FRACTION,
    DEFAULT_SIMPLE_STAGE_TIMEOUT,
    MAX_CODE_LENGTH,
)
from gigaevo.entrypoint.default_pipelines import DefaultPipelineBuilder
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.programs.dag.automata import ExecutionOrderDependency
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.stages.ancestry_selector import AncestrySelector
from gigaevo.programs.stages.collector import DescendantProgramIds
from gigaevo.programs.stages.lineage_memory import (
    ConcatMemoryStage,
    IntraMemoryStage,
)
from gigaevo.programs.stages.memory_context import MemoryContextStage
from gigaevo.programs.stages.mutation_suggestions import MutationSuggestionStage

DEFAULT_INTRA_MAX_CHILDREN = 24


class IntraMemoryPipelineBuilder(DefaultPipelineBuilder):
    """Default pipeline + DAG-native intra memory (no cross-population channel).

    Inherits from :class:`DefaultPipelineBuilder` (not the contextual variant)
    so problems without a ``context.py`` file (e.g. heilbron) are supported.

    Used by ``pipeline=standard``. The end-of-run external-memory extractor
    (IdeaTracker) is independent of this builder — opt in by passing
    ``ideas_tracker=default`` on the CLI; the cards it writes are available
    for subsequent runs that use ``pipeline=intra_extra_memory``, but they
    are NOT consumed by this builder's DAG.
    """

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 3600.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
        max_parallel: int | None = None,
        max_insights: int = 7,
        max_code_length: int = MAX_CODE_LENGTH,
        archive_gate_enabled: bool = False,
        intra_max_children: int = DEFAULT_INTRA_MAX_CHILDREN,
        mutation_mode: str | None = None,
        enable_optuna_stage: bool = False,
        optimization_time_budget: float | None = None,
    ):
        super().__init__(
            ctx,
            dag_timeout=dag_timeout,
            stage_timeout=stage_timeout,
            max_parallel=max_parallel,
            max_insights=max_insights,
            max_code_length=max_code_length,
            archive_gate_enabled=archive_gate_enabled,
        )
        self._enable_optuna_stage = enable_optuna_stage
        self._optimization_time_budget_arg = optimization_time_budget
        self._dag_timeout_arg = dag_timeout

        metrics_context = self.ctx.problem_ctx.metrics_context
        storage = self.ctx.storage
        strong_llm = self.ctx.llm_wrapper
        intra_max_children_val = intra_max_children
        task_description = self.ctx.problem_ctx.task_description
        mutation_mode_val = mutation_mode

        # Override the default DescendantProgramIds (which the default builder
        # configures with max_selected=1 for LineageStage) with a wider one
        # tailored to intra-memory analysis: the analyst LLM needs to see the
        # bulk of recent children, not just the single best.
        intra_descendant_selector = AncestrySelector(
            metrics_context=metrics_context,
            strategy="best_fitness",
            max_selected=intra_max_children_val,
        )
        self.replace_stage(
            "DescendantProgramIds",
            lambda: DescendantProgramIds(
                storage=storage,
                selector=intra_descendant_selector,
                timeout=stage_timeout,
            ),
        )

        self.add_stage(
            "IntraMemoryStage",
            lambda: IntraMemoryStage(
                llm=strong_llm,
                storage=storage,
                metrics_context=metrics_context,
                max_children=intra_max_children_val,
                task_description=task_description,
                timeout=stage_timeout,
            ),
        )

        # Captured for the MutationSuggestionStage factory below.
        max_insights_val = max_insights

        self.add_stage(
            "MutationSuggestionStage",
            lambda: MutationSuggestionStage(
                llm=strong_llm,
                storage=storage,
                metrics_context=metrics_context,
                task_description=task_description,
                max_insights=max_insights_val,
                timeout=stage_timeout,
                mutation_mode=mutation_mode_val,
            ),
        )

        # Strip legacy lineage stages — superseded by IntraMemoryStage
        # (per-parent lineage card via strong LLM) + MutationSuggestionStage
        # (prescriptive insights). DescendantProgramIds is INTENTIONALLY kept;
        # it now feeds IntraMemoryStage instead of LineagesToDescendants.
        # ``remove_stage`` drops the node, every edge touching it, and any
        # deps referencing it.
        #
        # MemoryContextStage is also removed: the intra-only DAG has no
        # consumer for cross-population cards (the suggester takes only the
        # intra card here, and the mutator's ``memory`` slot is fed directly
        # by the intra card). The IntraExtraMemoryPipelineBuilder subclass
        # re-adds it when the extra channel is needed.
        for legacy in (
            "AncestorProgramIds",
            "LineageStage",
            "LineagesToDescendants",
            "LineagesFromAncestors",
            "InsightsStage",
            "MemoryContextStage",
        ):
            self.remove_stage(legacy)

        # Rewire MutationContextStage with the intra-only descriptive/prescriptive split:
        #   * IntraMemoryStage is DESCRIPTIVE and consumes only ``children_ids``.
        #   * MutationSuggestionStage is PRESCRIPTIVE: it takes the intra card
        #     (+ ancestral trail walked from storage internally) and emits
        #     structured ``ProgramInsights`` into MutationContextStage's
        #     ``insights`` slot — same shape the legacy ``InsightsStage``
        #     produced, so the mutator's PROGRAM INSIGHTS section renders
        #     unchanged.
        #   * The intra card wires DIRECTLY into MutationContextStage.memory
        #     (Box[str] → StringContainer | None) — no ConcatMemoryStage
        #     needed when there's no second channel to join.
        self.add_data_flow_edge(
            "DescendantProgramIds", "IntraMemoryStage", "children_ids"
        )
        self.add_data_flow_edge(
            "IntraMemoryStage", "MutationSuggestionStage", "intra_card"
        )
        self.add_data_flow_edge(
            "EvolutionaryStatisticsCollector",
            "MutationSuggestionStage",
            "evolutionary_statistics",
        )
        self.add_data_flow_edge(
            "MutationSuggestionStage", "MutationContextStage", "insights"
        )
        self.add_data_flow_edge("IntraMemoryStage", "MutationContextStage", "memory")

        # Execution-order deps so intra fires after metrics + its upstream
        # collector, and MutationSuggestionStage fires after intra.
        #
        # Both strong-LLM stages (IntraMemoryStage AND MutationSuggestionStage)
        # are gated on validator success to mirror the old InsightsStage
        # skip-cascade — spending paid strong-LLM tokens on a program whose
        # code didn't even validate is pure waste. When the archive gate is
        # enabled, also gate on archive acceptance: a program that won't enter
        # the archive will never be a mutation parent, so neither its lineage
        # card nor its suggestions are ever read — skip both calls outright
        # (mirrors the legacy InsightsStage archive-gate dep at
        # default_pipelines.py L460-L463).
        self.add_exec_dep(
            "IntraMemoryStage",
            ExecutionOrderDependency.on_success("CallValidatorFunction"),
        )
        self.add_exec_dep(
            "IntraMemoryStage",
            ExecutionOrderDependency.always_after("EnsureMetricsStage"),
        )
        self.add_exec_dep(
            "IntraMemoryStage",
            ExecutionOrderDependency.always_after("DescendantProgramIds"),
        )
        # MutationSuggestionStage gates: validator-success + always-after both
        # data upstreams + EnsureMetricsStage (so the suggester sees finalised
        # metrics on the parent it analyses).
        self.add_exec_dep(
            "MutationSuggestionStage",
            ExecutionOrderDependency.on_success("CallValidatorFunction"),
        )
        self.add_exec_dep(
            "MutationSuggestionStage",
            ExecutionOrderDependency.always_after("EnsureMetricsStage"),
        )
        self.add_exec_dep(
            "MutationSuggestionStage",
            ExecutionOrderDependency.always_after("IntraMemoryStage"),
        )
        self.add_exec_dep(
            "MutationSuggestionStage",
            ExecutionOrderDependency.always_after("EvolutionaryStatisticsCollector"),
        )

        if self._archive_gate_enabled:
            # ArchivePotentialGateStage is configured by DefaultPipelineBuilder
            # with its own on_success(CallValidatorFunction). Both strong-LLM
            # stages piggyback on it so they skip-cascade when the gate
            # rejects — neither the descriptive card nor the prescriptive
            # suggestions are ever consumed for a rejected program.
            self.add_exec_dep(
                "IntraMemoryStage",
                ExecutionOrderDependency.on_success("ArchivePotentialGateStage"),
            )
            self.add_exec_dep(
                "MutationSuggestionStage",
                ExecutionOrderDependency.on_success("ArchivePotentialGateStage"),
            )

        if self._enable_optuna_stage:
            self._optimization_time_budget = (
                self._optimization_time_budget_arg
                if self._optimization_time_budget_arg is not None
                else self._dag_timeout_arg * DEFAULT_OPTIMIZATION_TIME_BUDGET_FRACTION
            )
            self._wire_optuna_stage()


class IntraExtraMemoryPipelineBuilder(IntraMemoryPipelineBuilder):
    """Intra base + extra (cross-population) memory channel.

    Used by ``pipeline=intra_extra_memory``. Re-adds the
    :class:`MemoryContextStage` that the intra-only base strips, plus
    :class:`ConcatMemoryStage` to join intra card + memory cards, and reroutes
    so the joined block (rather than the bare intra card) feeds
    ``MutationContextStage.memory``. The matching YAML wires
    :class:`LiveMemoryRefreshHook` into the engine's ``post_step_hook`` so the
    extra cards are refreshed mid-run.

    REQUIRED CLI co-overrides (the YAML cannot safely flip these from inside
    the ``pipeline/`` config group):

        ideas_tracker=default   — IdeaTracker is what LiveMemoryRefreshHook calls.
        memory=local            — MemorySelectorAgent reads the local card store
                                  that IdeaTracker writes to between refreshes.
        OPENROUTER_API_KEY=...  — GAM extra-memory agents call OpenRouter directly.

    Verify ``.hydra/config.yaml`` does not show ``Null*`` targets, and that
    ``/proc/<pid>/environ`` contains the OpenRouter key, before trusting
    extra-memory results.
    """

    def __init__(
        self,
        ctx: EvolutionContext,
        *,
        dag_timeout: float = 3600.0,
        stage_timeout: float = DEFAULT_SIMPLE_STAGE_TIMEOUT,
        max_parallel: int | None = None,
        max_insights: int = 7,
        max_code_length: int = MAX_CODE_LENGTH,
        archive_gate_enabled: bool = False,
        intra_max_children: int = DEFAULT_INTRA_MAX_CHILDREN,
        mutation_mode: str | None = None,
        enable_optuna_stage: bool = False,
        optimization_time_budget: float | None = None,
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
        )

        memory_provider = self.ctx.memory_provider
        task_description = self.ctx.problem_ctx.task_description
        metrics_context = self.ctx.problem_ctx.metrics_context
        metrics_description = MetricsFormatter(
            metrics_context
        ).format_metrics_description()

        # Re-add MemoryContextStage (the intra-only base strips it). Factory
        # signature mirrors DefaultPipelineBuilder._contribute_default_nodes.
        self.add_stage(
            "MemoryContextStage",
            lambda: MemoryContextStage(
                memory_provider=memory_provider,
                task_description=task_description,
                metrics_description=metrics_description,
                timeout=stage_timeout,
            ),
        )

        self.add_stage(
            "ConcatMemoryStage",
            lambda: ConcatMemoryStage(timeout=stage_timeout),
        )

        # Replace the intra-only base's direct IntraMemoryStage → MutationContextStage.memory
        # edge with the joined intra+memory_cards block via ConcatMemoryStage.
        self.remove_data_flow_edge("IntraMemoryStage", "MutationContextStage")
        self.add_data_flow_edge(
            "MemoryContextStage", "MutationSuggestionStage", "memory_cards"
        )
        self.add_data_flow_edge("IntraMemoryStage", "ConcatMemoryStage", "intra")
        self.add_data_flow_edge("MemoryContextStage", "ConcatMemoryStage", "cards")
        self.add_data_flow_edge("ConcatMemoryStage", "MutationContextStage", "memory")

        # Exec deps for the re-introduced stages.
        self.add_exec_dep(
            "MutationSuggestionStage",
            ExecutionOrderDependency.always_after("MemoryContextStage"),
        )
        self.add_exec_dep(
            "ConcatMemoryStage",
            ExecutionOrderDependency.always_after("IntraMemoryStage"),
        )
        self.add_exec_dep(
            "ConcatMemoryStage",
            ExecutionOrderDependency.always_after("MemoryContextStage"),
        )
