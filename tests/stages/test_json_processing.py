"""Tests for JSON processing stages: MergeDictStage, MergeStrFloatDict,
ParseJSONStage, StringifyJSONStage."""

from __future__ import annotations

from gigaevo.programs.core_types import StageState
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import Box
from gigaevo.programs.stages.json_processing import (
    MergeDictStage,
    MergeStrFloatDict,
    ParseJSONStage,
    StringifyJSONStage,
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
# TestParseJSONStage
# ---------------------------------------------------------------------------


class TestParseJSONStage:
    async def test_parse_object(self):
        """Valid JSON object parsed correctly."""
        stage = ParseJSONStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": '{"key": "value", "num": 42}'})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == {"key": "value", "num": 42}

    async def test_parse_array(self):
        """Valid JSON array parsed correctly."""
        stage = ParseJSONStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": "[1, 2, 3]"})
        result = await stage.execute(_prog())

        assert result.output.data == [1, 2, 3]

    async def test_parse_string(self):
        """JSON string value."""
        stage = ParseJSONStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": '"hello"'})
        result = await stage.execute(_prog())

        assert result.output.data == "hello"

    async def test_parse_number(self):
        """JSON number."""
        stage = ParseJSONStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": "42.5"})
        result = await stage.execute(_prog())

        assert result.output.data == 42.5

    async def test_parse_null(self):
        """JSON null → Python None."""
        stage = ParseJSONStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": "null"})
        result = await stage.execute(_prog())

        assert result.output.data is None

    async def test_parse_boolean(self):
        """JSON booleans."""
        stage = ParseJSONStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": "true"})
        result = await stage.execute(_prog())

        assert result.output.data is True

    async def test_invalid_json(self):
        """Invalid JSON → stage FAILED."""
        stage = ParseJSONStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": "{invalid json"})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert "Invalid JSON" in result.error.message

    async def test_empty_string(self):
        """Empty string → stage FAILED (json.loads('') raises)."""
        stage = ParseJSONStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": ""})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED


# ---------------------------------------------------------------------------
# TestStringifyJSONStage
# ---------------------------------------------------------------------------


class TestStringifyJSONStage:
    async def test_stringify_dict(self):
        """Dict → JSON string."""
        stage = StringifyJSONStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": {"key": "value"}})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert '"key"' in result.output.data
        assert '"value"' in result.output.data

    async def test_stringify_list(self):
        """List → JSON array string."""
        stage = StringifyJSONStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": [1, 2, 3]})
        result = await stage.execute(_prog())

        assert result.output.data == "[1, 2, 3]"

    async def test_stringify_with_indent(self):
        """indent=2 produces formatted output."""
        stage = StringifyJSONStage(indent=2, timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": {"a": 1}})
        result = await stage.execute(_prog())

        assert "\n" in result.output.data  # Indented output has newlines

    async def test_stringify_none(self):
        """None → 'null'."""
        stage = StringifyJSONStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": None})
        result = await stage.execute(_prog())

        assert result.output.data == "null"

    async def test_non_serializable_fails(self):
        """Non-serializable object → stage FAILED."""
        stage = StringifyJSONStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": object()})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert "Cannot convert to JSON" in result.error.message

    async def test_roundtrip(self):
        """Stringify then parse gives back original."""
        original = {"nested": {"a": [1, 2, 3]}, "key": "value"}
        # Stringify
        s_stage = StringifyJSONStage(timeout=5.0)
        s_stage.__class__.cache_handler = NO_CACHE
        s_stage.attach_inputs({"data": original})
        s_result = await s_stage.execute(_prog())
        json_str = s_result.output.data

        # Parse
        p_stage = ParseJSONStage(timeout=5.0)
        p_stage.__class__.cache_handler = NO_CACHE
        p_stage.attach_inputs({"data": json_str})
        p_result = await p_stage.execute(_prog())

        assert p_result.output.data == original


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
