"""Tests for JSON processing stages: MergeDictStage, MergeStrFloatDict."""

from __future__ import annotations

from gigaevo.programs.core_types import StageState
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import Box
from gigaevo.programs.stages.json_processing import (
    MergeDictStage,
    MergeStrFloatDict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog() -> Program:
    return Program(code="def solve(): return 42", state=ProgramState.RUNNING)


# ---------------------------------------------------------------------------
# TestMergeStrFloatDict
# ---------------------------------------------------------------------------


class TestMergeStrFloatDict:
    async def test_merge_no_overlap(self):
        """Two dicts with distinct keys → union."""
        stage = MergeStrFloatDict(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        first = Box[dict[str, float]](data={"a": 1.0})
        second = Box[dict[str, float]](data={"b": 2.0})
        stage.attach_inputs({"first": first, "second": second})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == {"a": 1.0, "b": 2.0}

    async def test_merge_with_overlap(self):
        """Overlapping key → second wins."""
        stage = MergeStrFloatDict(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        first = Box[dict[str, float]](data={"a": 1.0, "b": 2.0})
        second = Box[dict[str, float]](data={"b": 99.0, "c": 3.0})
        stage.attach_inputs({"first": first, "second": second})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data["b"] == 99.0  # second wins
        assert result.output.data["a"] == 1.0
        assert result.output.data["c"] == 3.0

    async def test_merge_empty_first(self):
        """Empty first dict → result equals second."""
        stage = MergeStrFloatDict(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        first = Box[dict[str, float]](data={})
        second = Box[dict[str, float]](data={"x": 5.0})
        stage.attach_inputs({"first": first, "second": second})
        result = await stage.execute(_prog())

        assert result.output.data == {"x": 5.0}

    async def test_merge_empty_second(self):
        """Empty second dict → result equals first."""
        stage = MergeStrFloatDict(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        first = Box[dict[str, float]](data={"x": 5.0})
        second = Box[dict[str, float]](data={})
        stage.attach_inputs({"first": first, "second": second})
        result = await stage.execute(_prog())

        assert result.output.data == {"x": 5.0}

    async def test_merge_both_empty(self):
        """Both empty → empty result."""
        stage = MergeStrFloatDict(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        first = Box[dict[str, float]](data={})
        second = Box[dict[str, float]](data={})
        stage.attach_inputs({"first": first, "second": second})
        result = await stage.execute(_prog())

        assert result.output.data == {}


# ---------------------------------------------------------------------------
# TestMergeDictStage
# ---------------------------------------------------------------------------


class TestMergeDictStage:
    async def test_basic_merge(self):
        """Basic merge of two dicts."""
        stage = MergeDictStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        first = Box(data={"a": 1, "b": 2})
        second = Box(data={"c": 3})
        stage.attach_inputs({"first": first, "second": second})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == {"a": 1, "b": 2, "c": 3}

    async def test_overlap_second_wins(self):
        """Overlapping keys → second overwrites first."""
        stage = MergeDictStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        first = Box(data={"key": "old"})
        second = Box(data={"key": "new"})
        stage.attach_inputs({"first": first, "second": second})
        result = await stage.execute(_prog())

        assert result.output.data["key"] == "new"
