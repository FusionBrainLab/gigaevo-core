"""Tests for ``IntraMemoryPipelineBuilder`` and ``IntraExtraMemoryPipelineBuilder``.

Two related pipeline builders share most of their DAG structure:

* ``IntraMemoryPipelineBuilder`` (intra-only, used by ``pipeline=standard``):
    * Per-parent lineage card via ``IntraMemoryStage``
    * Structured suggestions via ``MutationSuggestionStage``
    * NO cross-population memory cards (``MemoryContextStage`` is dropped)
    * NO ``ConcatMemoryStage`` — intra card wires straight to
      ``MutationContextStage.memory``

* ``IntraExtraMemoryPipelineBuilder`` (intra + extra; used by
  ``pipeline=intra_extra_memory``) is a subclass that re-adds
  ``MemoryContextStage`` and feeds the cross-population cards ONLY to
  ``MutationSuggestionStage`` (``memory_cards`` slot). The mutator's
  ``memory`` slot keeps the bare intra card, identical to the base.

Both drop all legacy lineage stages (``InsightsStage``, ``LineageStage``,
``AncestorProgramIds``, ``LineagesFromAncestors``, ``LineagesToDescendants``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from loguru import logger
import pytest

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.entrypoint.lineage_memory_pipeline import (
    IntraExtraMemoryPipelineBuilder,
    IntraMemoryPipelineBuilder,
)
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.core import MemorySelection
from gigaevo.memory.provider import MemoryProvider
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
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


def _make_ctx(memory_provider: MemoryProvider | None = None) -> EvolutionContext:
    metrics_ctx = _make_metrics_context()
    problem_ctx = MagicMock(spec=ProblemContext)
    problem_ctx.problem_dir = Path("/fake/problem")
    problem_ctx.task_description = "Solve the task."
    problem_ctx.metrics_context = metrics_ctx
    problem_ctx.is_contextual = False

    storage = MagicMock(spec=ProgramStorage)
    llm_wrapper = MagicMock(spec=MultiModelRouter)

    extra = {} if memory_provider is None else {"memory_provider": memory_provider}
    return EvolutionContext(
        problem_ctx=problem_ctx,
        llm_wrapper=llm_wrapper,
        storage=storage,
        prompts_dir=None,
        **extra,
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

    def test_concat_memory_stage_absent(self):
        """Cards reach the LLMs only via the suggester — there is no second
        verbatim channel to join, so ``ConcatMemoryStage`` must not exist."""
        bp = self._build()
        assert "ConcatMemoryStage" not in bp.nodes

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

    def test_memory_cards_feed_only_suggestion_stage(self):
        """``MemoryContextStage`` has exactly one consumer: the suggester.
        Cards must never reach the mutator verbatim."""
        bp = self._build()
        consumers = {
            dest for src, dest in _edge_src_dest(bp) if src == "MemoryContextStage"
        }
        assert consumers == {"MutationSuggestionStage"}

    def test_intra_card_feeds_mutation_context_memory_directly(self):
        """The mutator's ``memory`` slot carries the bare intra card, exactly
        like the intra-only base — cards are digested by the suggester into
        the ``insights`` slot instead of being concatenated verbatim."""
        bp = self._build()
        assert (
            "IntraMemoryStage",
            "MutationContextStage",
            "memory",
        ) in _edge_triples(bp)

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

    def test_intra_card_feeds_memory_context_stage(self):
        """GAM-fresh-context reorder: the selector is conditioned on the
        this-pass lineage card, so the intra card must feed MemoryContextStage."""
        bp = self._build()
        assert (
            "IntraMemoryStage",
            "MemoryContextStage",
            "intra_card",
        ) in _edge_triples(bp)

    def test_evolutionary_stats_feed_memory_context_stage(self):
        """GAM-fresh-context reorder: the selector also sees the live evolutionary
        snapshot (replaces the stale assembled mutation_context)."""
        bp = self._build()
        assert (
            "EvolutionaryStatisticsCollector",
            "MemoryContextStage",
            "evolutionary_statistics",
        ) in _edge_triples(bp)

    def test_reorder_off_drops_intra_card_edge_to_memory_context(self):
        """Arm B (``fresh_context_reorder=False``): the selector is NOT fed the
        this-pass lineage card, reverting to the pre-reorder DAG."""
        bp = self._build(fresh_context_reorder=False)
        assert (
            "IntraMemoryStage",
            "MemoryContextStage",
            "intra_card",
        ) not in _edge_triples(bp)

    def test_reorder_off_drops_evo_edge_to_memory_context(self):
        bp = self._build(fresh_context_reorder=False)
        assert (
            "EvolutionaryStatisticsCollector",
            "MemoryContextStage",
            "evolutionary_statistics",
        ) not in _edge_triples(bp)

    def test_reorder_off_keeps_base_extra_channel_wiring(self):
        """The toggle gates ONLY the two reorder edges — the base extra channel
        (cards → suggester) is untouched, so Arm B still injects cards."""
        bp = self._build(fresh_context_reorder=False)
        assert (
            "MemoryContextStage",
            "MutationSuggestionStage",
            "memory_cards",
        ) in _edge_triples(bp)

    def test_reorder_on_by_default(self):
        """Default is the shipping behaviour (reorder ON) — both reorder edges
        present without an explicit flag."""
        bp = self._build()
        triples = _edge_triples(bp)
        assert ("IntraMemoryStage", "MemoryContextStage", "intra_card") in triples
        assert (
            "EvolutionaryStatisticsCollector",
            "MemoryContextStage",
            "evolutionary_statistics",
        ) in triples


# ===================================================================
# Arm-mismatch warnings — read-side provider vs pipeline capability
# ===================================================================


class _CardProvider(MemoryProvider):
    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None = None,
    ) -> MemorySelection:
        return MemorySelection(cards=[], card_ids=[])


@pytest.fixture
def warnings_log() -> list[str]:
    messages: list[str] = []
    handle = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(handle)


def _arm_warnings(messages: list[str]) -> list[str]:
    return [m for m in messages if "[Memory][Arm]" in m]


class TestArmMismatchWarnings:
    def test_intra_extra_with_null_provider_warns_read_path_disabled(
        self, warnings_log
    ):
        # arm 2′ (write-cost-controlled baseline) is legitimate and currently
        # used live — this must stay a WARNING, never a raise.
        IntraExtraMemoryPipelineBuilder(_make_ctx())
        (warning,) = _arm_warnings(warnings_log)
        assert "read path DISABLED" in warning

    def test_intra_extra_with_real_provider_is_silent(self, warnings_log):
        IntraExtraMemoryPipelineBuilder(_make_ctx(_CardProvider()))
        assert _arm_warnings(warnings_log) == []

    def test_standard_with_real_provider_warns_cards_never_read(self, warnings_log):
        IntraMemoryPipelineBuilder(_make_ctx(_CardProvider()))
        (warning,) = _arm_warnings(warnings_log)
        assert "never reads" in warning

    def test_standard_with_null_provider_is_silent(self, warnings_log):
        IntraMemoryPipelineBuilder(_make_ctx())
        assert _arm_warnings(warnings_log) == []
