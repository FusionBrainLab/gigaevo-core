"""Tests for the GAM-fresh-context reorder.

The extra-memory card selector (``MemoryContextStage`` → GAM) used to build its
retrieval query from ``parent.metadata[MUTATION_CONTEXT]`` — the assembled
mutation context written by a *downstream* stage, hence one DAG pass stale (and
empty on a program's first pass). The reorder feeds the selector the **fresh
this-pass** lineage card + live evolutionary snapshot instead.

These tests pin the query-construction change ONLY. Nothing downstream of the
query (shortlist/auction/budget/renderer) is touched.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.memory.core import MemorySelection
from gigaevo.memory.core.card_selector import LLMCardSelector
from gigaevo.memory.provider import MemoryProvider, SelectorMemoryProvider
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.stages.collector import EvolutionaryStatistics
from gigaevo.programs.stages.common import StringContainer
from gigaevo.programs.stages.memory_context import MemoryContextStage


def _make_program(code: str = "def solve(): return 42") -> Program:
    return Program(code=code)


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
        }
    )


def _make_evo_stats() -> EvolutionaryStatistics:
    return EvolutionaryStatistics(
        generation=1,
        iteration=None,
        current_program_metrics={"fitness": 0.5},
        best_fitness={"fitness": 0.9},
        worst_fitness={"fitness": 0.1},
        average_fitness={"fitness": 0.5},
        valid_rate=1.0,
        total_program_count=5,
        avg_num_children=1.0,
        max_num_children=3,
        ancestor_count=1,
        best_fitness_in_ancestors={"fitness": 0.7},
        worst_fitness_in_ancestors={"fitness": 0.7},
        average_fitness_in_ancestors={"fitness": 0.7},
        valid_rate_in_ancestors=1.0,
        descendant_count=0,
        best_fitness_in_descendants={},
        worst_fitness_in_descendants={},
        average_fitness_in_descendants={},
        valid_rate_in_descendants=0.0,
    )


class _CaptureProvider(MemoryProvider):
    """Records the parent_context the stage hands to the selector boundary."""

    def __init__(self) -> None:
        self.parent_context: str | None = None
        self.called = False

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None = None,
    ) -> MemorySelection:
        self.called = True
        self.parent_context = parent_context
        return MemorySelection(cards=[], card_ids=[])


# ---------------------------------------------------------------------------
# Selector query construction
# ---------------------------------------------------------------------------


class TestSelectorParentContexts:
    def test_explicit_parent_contexts_used_over_stale_metadata(self) -> None:
        selector = LLMCardSelector()
        parent = _make_program()
        parent.set_metadata(MUTATION_CONTEXT_METADATA_KEY, "STALE-PRIOR-PASS-BLOCK")

        req = selector.build_core_request(
            parents=[parent],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
            max_cards=2,
            parent_contexts=["FRESH-LINEAGE-CARD-AND-EVO"],
        )

        assert "FRESH-LINEAGE-CARD-AND-EVO" in req
        assert "STALE-PRIOR-PASS-BLOCK" not in req

    def test_falls_back_to_metadata_without_parent_contexts(self) -> None:
        # Backward compat: callers that don't pass parent_contexts (other
        # pipelines, direct unit calls) keep the legacy metadata behaviour.
        selector = LLMCardSelector()
        parent = _make_program()
        parent.set_metadata(MUTATION_CONTEXT_METADATA_KEY, "STALE-PRIOR-PASS-BLOCK")

        req = selector.build_core_request(
            parents=[parent],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
            max_cards=2,
        )

        assert "STALE-PRIOR-PASS-BLOCK" in req

    def test_parent_code_always_present(self) -> None:
        selector = LLMCardSelector()
        parent = _make_program(code="def unique_marker(): return 1")
        req = selector.build_core_request(
            parents=[parent],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
            max_cards=1,
            parent_contexts=["ctx"],
        )
        assert "def unique_marker(): return 1" in req


# ---------------------------------------------------------------------------
# Stage assembles the fresh context from upstream artifacts
# ---------------------------------------------------------------------------


class TestStageAssemblesFreshContext:
    @pytest.mark.asyncio
    async def test_intra_card_and_evo_reach_the_selector(self) -> None:
        provider = _CaptureProvider()
        stage = MemoryContextStage(
            memory_provider=provider,
            task_description="t",
            metrics_description="m",
            metrics_context=_make_metrics_context(),
            timeout=60,
        )
        stage.attach_inputs(
            {
                "intra_card": StringContainer(
                    data="## Intra Memory — Per-Parent Lineage Card\n\nLINEAGE-BODY-XYZ"
                ),
                "evolutionary_statistics": _make_evo_stats(),
            }
        )

        await stage.compute(_make_program())

        assert provider.called
        assert "LINEAGE-BODY-XYZ" in provider.parent_context
        assert "## Evolutionary Statistics" in provider.parent_context

    @pytest.mark.asyncio
    async def test_cold_seed_passes_empty_context(self) -> None:
        # No intra card, no evo snapshot (a freshly-born program's first pass):
        # the selector gets code only, never a stale block.
        provider = _CaptureProvider()
        stage = MemoryContextStage(
            memory_provider=provider,
            task_description="t",
            metrics_description="m",
            metrics_context=_make_metrics_context(),
            timeout=60,
        )

        await stage.compute(_make_program())

        assert provider.called
        assert provider.parent_context == ""

    @pytest.mark.asyncio
    async def test_evo_skipped_without_metrics_context(self) -> None:
        # metrics_context is optional; without it the evo snapshot cannot be
        # rendered, so only the intra card flows (graceful degrade).
        provider = _CaptureProvider()
        stage = MemoryContextStage(
            memory_provider=provider,
            task_description="t",
            metrics_description="m",
            timeout=60,
        )
        stage.attach_inputs(
            {
                "intra_card": StringContainer(data="LINEAGE-ONLY"),
                "evolutionary_statistics": _make_evo_stats(),
            }
        )

        await stage.compute(_make_program())

        assert provider.parent_context == "LINEAGE-ONLY"


# ---------------------------------------------------------------------------
# Provider threads parent_context to the pipeline query layer
# ---------------------------------------------------------------------------


class TestFreshContextReorderToggle:
    """The reorder is gated so an A/B can run both arms off one binary.

    ``fresh_context_reorder=False`` (Arm B) makes the stage pass ``None`` so the
    selector falls back to ``parent.metadata[MUTATION_CONTEXT]`` — the exact
    pre-reorder behaviour. ``True`` (default, Arm C) builds the fresh context.
    """

    @pytest.mark.asyncio
    async def test_reorder_off_passes_none_for_metadata_fallback(self) -> None:
        provider = _CaptureProvider()
        stage = MemoryContextStage(
            memory_provider=provider,
            task_description="t",
            metrics_description="m",
            metrics_context=_make_metrics_context(),
            timeout=60,
            fresh_context_reorder=False,
        )
        stage.attach_inputs(
            {
                "intra_card": StringContainer(data="LINEAGE-BODY-XYZ"),
                "evolutionary_statistics": _make_evo_stats(),
            }
        )

        await stage.compute(_make_program())

        assert provider.called
        # None => the selector falls back to the stale metadata block (Arm B).
        assert provider.parent_context is None

    @pytest.mark.asyncio
    async def test_reorder_on_is_default_and_builds_fresh(self) -> None:
        provider = _CaptureProvider()
        stage = MemoryContextStage(
            memory_provider=provider,
            task_description="t",
            metrics_description="m",
            metrics_context=_make_metrics_context(),
            timeout=60,
        )
        stage.attach_inputs(
            {
                "intra_card": StringContainer(data="LINEAGE-BODY-XYZ"),
                "evolutionary_statistics": _make_evo_stats(),
            }
        )

        await stage.compute(_make_program())

        assert "LINEAGE-BODY-XYZ" in provider.parent_context


class TestProviderThreadsParentContext:
    @pytest.mark.asyncio
    async def test_selector_provider_passes_parent_contexts_list(self) -> None:
        mock_pipeline = AsyncMock()
        mock_pipeline.select.return_value = MemorySelection(cards=[], card_ids=[])
        provider = SelectorMemoryProvider(backend=lambda **_kw: None, max_cards=3)
        provider._pipeline = mock_pipeline

        await provider.select_cards(
            _make_program(),
            task_description="t",
            metrics_description="m",
            parent_context="FRESH-CTX",
        )

        _, kwargs = mock_pipeline.select.call_args
        assert kwargs["parent_contexts"] == ["FRESH-CTX"]

    @pytest.mark.asyncio
    async def test_none_parent_context_threads_none(self) -> None:
        mock_pipeline = AsyncMock()
        mock_pipeline.select.return_value = MemorySelection(cards=[], card_ids=[])
        provider = SelectorMemoryProvider(backend=lambda **_kw: None, max_cards=3)
        provider._pipeline = mock_pipeline

        await provider.select_cards(
            _make_program(), task_description="t", metrics_description="m"
        )

        _, kwargs = mock_pipeline.select.call_args
        assert kwargs["parent_contexts"] is None
