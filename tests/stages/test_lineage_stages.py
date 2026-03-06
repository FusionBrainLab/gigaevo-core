"""Tests for LineagesToDescendants and LineagesFromAncestors stages."""

from __future__ import annotations

from unittest.mock import AsyncMock

from gigaevo.llm.agents.lineage import (
    TransitionAnalysis,
    TransitionInsight,
    TransitionInsights,
)
from gigaevo.programs.core_types import ProgramStageResult, StageState
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import ListOf
from gigaevo.programs.stages.insights_lineage import (
    LineageAnalysesOutput,
    LineagesFromAncestors,
    LineagesToDescendants,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog() -> Program:
    return Program(code="def solve(): return 42", state=ProgramState.RUNNING)


def _make_insights() -> TransitionInsights:
    return TransitionInsights(
        insights=[
            TransitionInsight(strategy="imitation", description="Copied pattern"),
            TransitionInsight(strategy="avoidance", description="Avoided pitfall"),
            TransitionInsight(strategy="exploration", description="New approach"),
        ]
    )


def _make_analysis(from_id: str, to_id: str) -> TransitionAnalysis:
    return TransitionAnalysis(
        from_id=from_id,
        to_id=to_id,
        parent_metrics={"score": 50.0},
        child_metrics={"score": 70.0},
        diff_blocks=["+ new line"],
        insights=_make_insights(),
    )


# ---------------------------------------------------------------------------
# TestLineagesToDescendants
# ---------------------------------------------------------------------------


class TestLineagesToDescendants:
    async def test_empty_child_ids_returns_skipped(self):
        """Empty descendant_ids → SKIPPED."""
        storage = AsyncMock()
        stage = LineagesToDescendants(
            storage=storage,
            source_stage_name="lineage_analysis",
            timeout=5.0,
        )
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"descendant_ids": ListOf[str](items=[])})
        result = await stage.execute(_prog())

        assert result.status == StageState.SKIPPED

    async def test_no_matching_analyses_returns_empty(self):
        """Children exist but have no result for source_stage_name → empty list."""
        storage = AsyncMock()
        child = _prog()
        child.stage_results = {}  # no results at all
        storage.mget.return_value = [child]

        stage = LineagesToDescendants(
            storage=storage,
            source_stage_name="lineage_analysis",
            timeout=5.0,
        )
        stage.__class__.cache_handler = NO_CACHE

        parent = _prog()
        stage.attach_inputs({"descendant_ids": ListOf[str](items=[child.id])})
        result = await stage.execute(parent)

        assert result.status == StageState.COMPLETED
        assert result.output.items == []

    async def test_correct_analysis_extracted(self):
        """Child has analysis for this parent → correct TransitionAnalysis returned."""
        storage = AsyncMock()
        parent = _prog()
        child = _prog()

        # Child has a LineageAnalysesOutput with analyses for parent→child
        analysis = _make_analysis(from_id=parent.id, to_id=child.id)
        # Also has an unrelated analysis from another parent
        other_analysis = _make_analysis(from_id="other-parent", to_id=child.id)
        output = LineageAnalysesOutput(analyses=[other_analysis, analysis])

        child.stage_results["lineage_analysis"] = ProgramStageResult.success(
            output=output
        )
        storage.mget.return_value = [child]

        stage = LineagesToDescendants(
            storage=storage,
            source_stage_name="lineage_analysis",
            timeout=5.0,
        )
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"descendant_ids": ListOf[str](items=[child.id])})
        result = await stage.execute(parent)

        assert result.status == StageState.COMPLETED
        assert len(result.output.items) == 1
        assert result.output.items[0].from_id == parent.id
        assert result.output.items[0].to_id == child.id

    async def test_no_analysis_for_this_parent(self):
        """Child has analyses but not for this parent → empty list."""
        storage = AsyncMock()
        parent = _prog()
        child = _prog()

        other_analysis = _make_analysis(from_id="other-parent", to_id=child.id)
        output = LineageAnalysesOutput(analyses=[other_analysis])
        child.stage_results["lineage_analysis"] = ProgramStageResult.success(
            output=output
        )
        storage.mget.return_value = [child]

        stage = LineagesToDescendants(
            storage=storage,
            source_stage_name="lineage_analysis",
            timeout=5.0,
        )
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"descendant_ids": ListOf[str](items=[child.id])})
        result = await stage.execute(parent)

        assert result.status == StageState.COMPLETED
        assert result.output.items == []

    async def test_child_with_null_output_skipped(self):
        """Child has result but output is None → skipped in iteration."""
        storage = AsyncMock()
        parent = _prog()
        child = _prog()
        child.stage_results["lineage_analysis"] = ProgramStageResult.success(
            output=None
        )
        storage.mget.return_value = [child]

        stage = LineagesToDescendants(
            storage=storage,
            source_stage_name="lineage_analysis",
            timeout=5.0,
        )
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"descendant_ids": ListOf[str](items=[child.id])})
        result = await stage.execute(parent)

        assert result.status == StageState.COMPLETED
        assert result.output.items == []


# ---------------------------------------------------------------------------
# TestLineagesFromAncestors
# ---------------------------------------------------------------------------


class TestLineagesFromAncestors:
    async def test_empty_parent_ids_returns_skipped(self):
        """Empty ancestor_ids → SKIPPED."""
        storage = AsyncMock()
        stage = LineagesFromAncestors(
            storage=storage,
            source_stage_name="lineage_analysis",
            timeout=5.0,
        )
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"ancestor_ids": ListOf[str](items=[])})
        result = await stage.execute(_prog())

        assert result.status == StageState.SKIPPED

    async def test_no_source_result_returns_skipped(self):
        """Current program has no result for source_stage_name → SKIPPED."""
        storage = AsyncMock()
        prog = _prog()
        prog.stage_results = {}

        stage = LineagesFromAncestors(
            storage=storage,
            source_stage_name="lineage_analysis",
            timeout=5.0,
        )
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"ancestor_ids": ListOf[str](items=["parent1"])})
        result = await stage.execute(prog)

        assert result.status == StageState.SKIPPED

    async def test_correct_analysis_filtered(self):
        """Program has analyses from multiple parents → only matching ones returned."""
        storage = AsyncMock()
        prog = _prog()

        parent1_id = "parent-1"
        parent2_id = "parent-2"
        unrelated_id = "other-parent"

        a1 = _make_analysis(from_id=parent1_id, to_id=prog.id)
        a2 = _make_analysis(from_id=parent2_id, to_id=prog.id)
        a_other = _make_analysis(from_id=unrelated_id, to_id=prog.id)

        output = LineageAnalysesOutput(analyses=[a1, a2, a_other])
        prog.stage_results["lineage_analysis"] = ProgramStageResult.success(
            output=output
        )

        stage = LineagesFromAncestors(
            storage=storage,
            source_stage_name="lineage_analysis",
            timeout=5.0,
        )
        stage.__class__.cache_handler = NO_CACHE
        # Only request parent1 and parent2, not unrelated
        stage.attach_inputs(
            {"ancestor_ids": ListOf[str](items=[parent1_id, parent2_id])}
        )
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert len(result.output.items) == 2
        from_ids = {a.from_id for a in result.output.items}
        assert from_ids == {parent1_id, parent2_id}

    async def test_no_matching_ancestor_returns_empty(self):
        """Program has analyses from other parents, not the requested one → empty list."""
        storage = AsyncMock()
        prog = _prog()

        a_other = _make_analysis(from_id="unrelated-parent", to_id=prog.id)
        output = LineageAnalysesOutput(analyses=[a_other])
        prog.stage_results["lineage_analysis"] = ProgramStageResult.success(
            output=output
        )

        stage = LineagesFromAncestors(
            storage=storage,
            source_stage_name="lineage_analysis",
            timeout=5.0,
        )
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"ancestor_ids": ListOf[str](items=["requested-parent"])})
        result = await stage.execute(prog)

        assert result.status == StageState.COMPLETED
        assert result.output.items == []

    async def test_null_output_returns_skipped(self):
        """Source stage result has None output → SKIPPED."""
        storage = AsyncMock()
        prog = _prog()
        prog.stage_results["lineage_analysis"] = ProgramStageResult.success(output=None)

        stage = LineagesFromAncestors(
            storage=storage,
            source_stage_name="lineage_analysis",
            timeout=5.0,
        )
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"ancestor_ids": ListOf[str](items=["parent1"])})
        result = await stage.execute(prog)

        assert result.status == StageState.SKIPPED
