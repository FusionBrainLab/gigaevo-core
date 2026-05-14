"""Tests for LineageStage instrumentation (baseline logging) and the
per-parent skip optimisation for failed children."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from loguru import logger
import pytest

from gigaevo.programs.core_types import ProgramStageResult, StageState
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.ancestry_selector import AncestrySelector
from gigaevo.programs.stages.common import CacheOnlyInput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PRIMARY = "score"


def _make_metrics_context(higher_is_better: bool = True) -> MetricsContext:
    return MetricsContext.from_descriptions(
        primary_key=_PRIMARY,
        primary_description="primary",
        higher_is_better=higher_is_better,
    )


def _make_program(
    score: float | None = None,
    failed: bool = False,
    parent_ids: list[str] | None = None,
    child_ids: list[str] | None = None,
) -> Program:
    prog = Program(code="def solve(): return 42", state=ProgramState.RUNNING)
    if score is not None:
        prog.metrics = {_PRIMARY: score, "is_valid": 1.0}
    if failed:
        prog.stage_results["ValidateCodeStage"] = ProgramStageResult(
            status=StageState.FAILED
        )
    if parent_ids:
        prog.lineage.parents = list(parent_ids)
    if child_ids:
        prog.lineage.children = list(child_ids)
    return prog


def _make_stage(storage, descendant_selector=None):
    """Bypass __init__ — only preprocess() needs storage + descendant_selector."""
    from gigaevo.programs.stages.insights_lineage import LineageStage

    stage = LineageStage.__new__(LineageStage)
    stage.storage = storage
    stage.descendant_selector = descendant_selector
    return stage


def _storage_mget_router(programs_by_id: dict[str, Program]):
    """Return an mget that resolves ids via the provided id->Program map."""

    async def _mget(ids, *args, **kwargs):
        return [programs_by_id[i] for i in ids if i in programs_by_id]

    return AsyncMock(side_effect=_mget)


# ---------------------------------------------------------------------------
# Baseline logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lineage_stage_logs_n_parents():
    """Base LineageStage logs program id and n_parents on every invocation."""
    storage = MagicMock()
    storage.mget = AsyncMock(
        return_value=[MagicMock(spec=Program), MagicMock(spec=Program)]
    )

    stage = _make_stage(storage)

    program = MagicMock(spec=Program)
    program.id = "abcdef1234-child"
    program.is_failed = False
    program.lineage = MagicMock()
    program.lineage.parents = ["parent-1", "parent-2"]

    messages: list[str] = []
    sink_id = logger.add(
        lambda m: messages.append(m.record["message"]),
        level="INFO",
    )
    try:
        result = await stage.preprocess(program, CacheOnlyInput())
    finally:
        logger.remove(sink_id)

    assert any("[LineageStage]" in msg and "n_parents=2" in msg for msg in messages), (
        f"Expected [LineageStage] n_parents=2 log, got {messages}"
    )
    assert isinstance(result, dict)
    assert len(result["parents"]) == 2


# ---------------------------------------------------------------------------
# Skip optimisation (failed child)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_no_op_when_program_succeeds():
    """Successful child → every parent is kept even with a selector wired."""
    mc = _make_metrics_context(higher_is_better=True)
    sibling = _make_program(score=99.0)
    parent = _make_program(score=10.0, child_ids=[sibling.id])
    program = _make_program(score=50.0, failed=False, parent_ids=[parent.id])
    parent.lineage.children = [sibling.id, program.id]

    storage = MagicMock()
    storage.mget = _storage_mget_router(
        {parent.id: parent, sibling.id: sibling, program.id: program}
    )
    selector = AncestrySelector(
        metrics_context=mc, strategy="best_fitness", max_selected=1
    )
    stage = _make_stage(storage, descendant_selector=selector)

    result = await stage.preprocess(program, CacheOnlyInput())

    assert [p.id for p in result["parents"]] == [parent.id]


@pytest.mark.asyncio
async def test_skip_no_op_when_selector_none():
    """Failed child but no selector wired → original behaviour, every parent kept."""
    parent_a = _make_program(score=10.0)
    parent_b = _make_program(score=20.0)
    program = _make_program(
        score=0.0, failed=True, parent_ids=[parent_a.id, parent_b.id]
    )

    storage = MagicMock()
    storage.mget = _storage_mget_router({parent_a.id: parent_a, parent_b.id: parent_b})
    stage = _make_stage(storage, descendant_selector=None)

    result = await stage.preprocess(program, CacheOnlyInput())

    assert {p.id for p in result["parents"]} == {parent_a.id, parent_b.id}


@pytest.mark.asyncio
async def test_skip_when_failed_and_better_sibling_exists():
    """Failed P with a strictly better sibling → (Q→P) skipped."""
    mc = _make_metrics_context(higher_is_better=True)
    sibling = _make_program(score=99.0)
    program = _make_program(score=0.0, failed=True)
    parent = _make_program(child_ids=[sibling.id, program.id])
    program.lineage.parents = [parent.id]

    storage = MagicMock()
    storage.mget = _storage_mget_router(
        {parent.id: parent, sibling.id: sibling, program.id: program}
    )
    selector = AncestrySelector(
        metrics_context=mc, strategy="best_fitness", max_selected=1
    )
    stage = _make_stage(storage, descendant_selector=selector)

    result = await stage.preprocess(program, CacheOnlyInput())

    assert result["parents"] == []


@pytest.mark.asyncio
async def test_keep_when_failed_but_no_better_siblings():
    """Failed P is parent Q's only child → (Q→P) kept (selector picks P)."""
    mc = _make_metrics_context(higher_is_better=True)
    program = _make_program(score=0.0, failed=True)
    parent = _make_program(child_ids=[program.id])
    program.lineage.parents = [parent.id]

    storage = MagicMock()
    storage.mget = _storage_mget_router({parent.id: parent, program.id: program})
    selector = AncestrySelector(
        metrics_context=mc, strategy="best_fitness", max_selected=1
    )
    stage = _make_stage(storage, descendant_selector=selector)

    result = await stage.preprocess(program, CacheOnlyInput())

    assert [p.id for p in result["parents"]] == [parent.id]


@pytest.mark.asyncio
async def test_skip_per_parent_independent():
    """Failed P, two parents: Q1 has a better sibling, Q2 only has P → only Q2 kept."""
    mc = _make_metrics_context(higher_is_better=True)
    program = _make_program(score=0.0, failed=True)
    better_sibling = _make_program(score=99.0)
    parent_with_better = _make_program(child_ids=[better_sibling.id, program.id])
    parent_solo = _make_program(child_ids=[program.id])
    program.lineage.parents = [parent_with_better.id, parent_solo.id]

    storage = MagicMock()
    storage.mget = _storage_mget_router(
        {
            parent_with_better.id: parent_with_better,
            parent_solo.id: parent_solo,
            better_sibling.id: better_sibling,
            program.id: program,
        }
    )
    selector = AncestrySelector(
        metrics_context=mc, strategy="best_fitness", max_selected=1
    )
    stage = _make_stage(storage, descendant_selector=selector)

    result = await stage.preprocess(program, CacheOnlyInput())

    assert [p.id for p in result["parents"]] == [parent_solo.id]


@pytest.mark.asyncio
async def test_failed_root_no_parents_noop():
    """Failed program with no parents → empty parents list, no errors."""
    mc = _make_metrics_context(higher_is_better=True)
    program = _make_program(score=0.0, failed=True)

    storage = MagicMock()
    storage.mget = _storage_mget_router({})
    selector = AncestrySelector(
        metrics_context=mc, strategy="best_fitness", max_selected=1
    )
    stage = _make_stage(storage, descendant_selector=selector)

    result = await stage.preprocess(program, CacheOnlyInput())

    assert result["parents"] == []
