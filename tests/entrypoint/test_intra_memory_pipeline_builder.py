"""Tests for ``IntraMemoryPipelineBuilder`` and ``IntraExtraMemoryPipelineBuilder``.

Two related pipeline builders share most of their DAG structure:

* ``IntraMemoryPipelineBuilder`` (intra-only, used by ``pipeline=standard``):
    * Per-parent lineage card via ``IntraMemoryStage``
    * Structured suggestions via ``MutationSuggestionStage``
    * NO cross-population memory cards (``MemoryContextStage`` is dropped)
    * NO ``ConcatMemoryStage`` — intra card wires straight to
      ``MutationContextStage.memory``

* ``IntraExtraMemoryPipelineBuilder`` (intra + extra; used by
  ``pipeline=intra_extra_memory``) is a subclass that re-adds the extra
  channel: ``MemoryContextStage`` + ``ConcatMemoryStage`` joining both
  blocks into ``MutationContextStage.memory``.

Both drop all legacy lineage stages (``InsightsStage``, ``LineageStage``,
``AncestorProgramIds``, ``LineagesFromAncestors``, ``LineagesToDescendants``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.entrypoint.lineage_memory_pipeline import (
    IntraExtraMemoryPipelineBuilder,
    IntraMemoryPipelineBuilder,
)
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.runner.dag_blueprint import DAGBlueprint

LEGACY_STAGES = (
    "InsightsStage",
    "AncestorProgramIds",
    "LineageStage",
    "LineagesFromAncestors",
    "LineagesToDescendants",
)


def _make_metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="main metric",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            "is_valid": MetricSpec(
                description="validity flag",
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        }
    )


def _make_ctx() -> EvolutionContext:
    metrics_ctx = _make_metrics_context()
    problem_ctx = MagicMock(spec=ProblemContext)
    problem_ctx.problem_dir = Path("/fake/problem")
    problem_ctx.task_description = "Solve the task."
    problem_ctx.metrics_context = metrics_ctx
    problem_ctx.is_contextual = False

    storage = MagicMock(spec=ProgramStorage)
    llm_wrapper = MagicMock(spec=MultiModelRouter)

    return EvolutionContext(
        problem_ctx=problem_ctx,
        llm_wrapper=llm_wrapper,
        storage=storage,
        prompts_dir=None,
    )


def _edge_triples(bp: DAGBlueprint) -> set[tuple[str, str, str]]:
    """Extract (source, dest, input_name) triples from data-flow edges."""
    return {
        (e.source_stage, e.destination_stage, e.input_name) for e in bp.data_flow_edges
    }


def _edge_src_dest(bp: DAGBlueprint) -> set[tuple[str, str]]:
    return {(e.source_stage, e.destination_stage) for e in bp.data_flow_edges}


# ===================================================================
# IntraMemoryPipelineBuilder — intra-only (pipeline=standard)
# ===================================================================


class TestIntraMemoryPipelineBuilder:
    def _build(self, **kwargs) -> DAGBlueprint:
        builder = IntraMemoryPipelineBuilder(_make_ctx(), **kwargs)
        return builder.build_blueprint()

    def test_intra_memory_stage_present(self):
        bp = self._build()
        assert "IntraMemoryStage" in bp.nodes

    def test_mutation_suggestion_stage_present(self):
        bp = self._build()
        assert "MutationSuggestionStage" in bp.nodes

    def test_descendant_program_ids_kept(self):
        """``DescendantProgramIds`` feeds ``IntraMemoryStage``; must stay."""
        bp = self._build()
        assert "DescendantProgramIds" in bp.nodes

    def test_mutation_context_stage_present(self):
        bp = self._build()
        assert "MutationContextStage" in bp.nodes

    def test_legacy_stages_dropped(self):
        bp = self._build()
        for stage in LEGACY_STAGES:
            assert stage not in bp.nodes, (
                f"Legacy stage {stage!r} should be removed in the intra-only "
                "builder (superseded by IntraMemoryStage + MutationSuggestionStage)"
            )

    def test_concat_memory_stage_absent(self):
        """Intra-only pipeline does NOT use ``ConcatMemoryStage``; the intra
        card wires straight into ``MutationContextStage.memory``."""
        bp = self._build()
        assert "ConcatMemoryStage" not in bp.nodes

    def test_memory_context_stage_absent(self):
        """Intra-only mode fully drops the extra channel — ``MemoryContextStage``
        is the source of cross-population cards we don't consume here."""
        bp = self._build()
        assert "MemoryContextStage" not in bp.nodes

    def test_descendants_feed_intra_memory(self):
        bp = self._build()
        assert (
            "DescendantProgramIds",
            "IntraMemoryStage",
            "children_ids",
        ) in _edge_triples(bp)

    def test_intra_card_feeds_suggestion_stage(self):
        bp = self._build()
        assert (
            "IntraMemoryStage",
            "MutationSuggestionStage",
            "intra_card",
        ) in _edge_triples(bp)

    def test_suggestions_feed_mutation_context_insights(self):
        bp = self._build()
        assert (
            "MutationSuggestionStage",
            "MutationContextStage",
            "insights",
        ) in _edge_triples(bp)

    def test_intra_card_feeds_mutation_context_memory_directly(self):
        """In intra-only mode, ``IntraMemoryStage``'s ``StringContainer`` feeds
        ``MutationContextStage.memory`` directly — no ``ConcatMemoryStage``
        is needed because there's no second channel to join."""
        bp = self._build()
        assert (
            "IntraMemoryStage",
            "MutationContextStage",
            "memory",
        ) in _edge_triples(bp)

    def test_evolutionary_stats_feed_suggestion_stage(self):
        bp = self._build()
        assert (
            "EvolutionaryStatisticsCollector",
            "MutationSuggestionStage",
            "evolutionary_statistics",
        ) in _edge_triples(bp)

    def test_no_memory_cards_edge_to_suggestion_stage(self):
        """``MemoryContextStage`` is removed, so its old edge into the suggester
        must not be re-added by anything else."""
        bp = self._build()
        assert (
            "MemoryContextStage",
            "MutationSuggestionStage",
        ) not in _edge_src_dest(bp)

    def test_no_memory_context_to_mutation_context_edge(self):
        """The default builder wires ``MemoryContextStage → MutationContextStage.memory``.
        Intra-only mode reroutes that slot to the intra card, so the original
        edge must be removed."""
        bp = self._build()
        assert (
            "MemoryContextStage",
            "MutationContextStage",
        ) not in _edge_src_dest(bp)


# ===================================================================
# IntraExtraMemoryPipelineBuilder — intra + extra (subclass)
# ===================================================================


class TestIntraExtraMemoryPipelineBuilder:
    """Regression tests for ``pipeline=intra_extra_memory`` wiring.

    These pin the wiring exercised by all extant cycle-N intra+extra runs so
    that the new ``IntraMemoryPipelineBuilder`` split doesn't accidentally
    drop edges from the extra-channel-enabled variant.
    """

    def _build(self, **kwargs) -> DAGBlueprint:
        builder = IntraExtraMemoryPipelineBuilder(_make_ctx(), **kwargs)
        return builder.build_blueprint()

    def test_subclasses_intra_memory_builder(self):
        """The intra+extra builder is a subclass — both share most wiring."""
        assert issubclass(IntraExtraMemoryPipelineBuilder, IntraMemoryPipelineBuilder)

    def test_intra_memory_stage_present(self):
        bp = self._build()
        assert "IntraMemoryStage" in bp.nodes

    def test_mutation_suggestion_stage_present(self):
        bp = self._build()
        assert "MutationSuggestionStage" in bp.nodes

    def test_concat_memory_stage_present(self):
        """Intra+extra variant keeps ``ConcatMemoryStage`` to join both channels."""
        bp = self._build()
        assert "ConcatMemoryStage" in bp.nodes

    def test_memory_context_stage_present(self):
        bp = self._build()
        assert "MemoryContextStage" in bp.nodes

    def test_legacy_stages_dropped(self):
        bp = self._build()
        for stage in LEGACY_STAGES:
            assert stage not in bp.nodes

    def test_memory_cards_feed_suggestion_stage(self):
        bp = self._build()
        assert (
            "MemoryContextStage",
            "MutationSuggestionStage",
            "memory_cards",
        ) in _edge_triples(bp)

    def test_intra_feeds_concat(self):
        bp = self._build()
        assert (
            "IntraMemoryStage",
            "ConcatMemoryStage",
            "intra",
        ) in _edge_triples(bp)

    def test_memory_cards_feed_concat(self):
        bp = self._build()
        assert (
            "MemoryContextStage",
            "ConcatMemoryStage",
            "cards",
        ) in _edge_triples(bp)

    def test_concat_feeds_mutation_context_memory(self):
        bp = self._build()
        assert (
            "ConcatMemoryStage",
            "MutationContextStage",
            "memory",
        ) in _edge_triples(bp)

    def test_no_direct_intra_to_mutation_context_memory(self):
        """The intra+extra variant routes through ``ConcatMemoryStage`` — the
        direct ``IntraMemoryStage → MutationContextStage.memory`` edge that the
        intra-only base wires must be removed in the subclass."""
        bp = self._build()
        assert (
            "IntraMemoryStage",
            "MutationContextStage",
            "memory",
        ) not in _edge_triples(bp)

    def test_no_memory_context_to_mutation_context_direct_edge(self):
        """The default builder wires ``MemoryContextStage → MutationContextStage.memory``.
        Both intra variants reroute that slot to alternative sources, so the
        direct edge must be removed in this variant too."""
        bp = self._build()
        assert (
            "MemoryContextStage",
            "MutationContextStage",
        ) not in _edge_src_dest(bp)

    def test_evolutionary_stats_feed_suggestion_stage(self):
        bp = self._build()
        assert (
            "EvolutionaryStatisticsCollector",
            "MutationSuggestionStage",
            "evolutionary_statistics",
        ) in _edge_triples(bp)
