"""Tests for FormatterStage."""

from __future__ import annotations

from typing import Any

from gigaevo.programs.core_types import StageState
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import AnyContainer
from gigaevo.programs.stages.formatter import FormatterStage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog() -> Program:
    return Program(code="def solve(): return 42", state=ProgramState.RUNNING)


# ---------------------------------------------------------------------------
# TestFormatterStage
# ---------------------------------------------------------------------------


class TestFormatterStage:
    async def test_none_input_returns_skipped(self):
        """None data → ProgramStageResult.skipped."""
        stage = FormatterStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": AnyContainer(data=None)})
        result = await stage.execute(_prog())

        assert result.status == StageState.SKIPPED

    async def test_string_passthrough(self):
        """String input → returned as-is."""
        stage = FormatterStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": AnyContainer(data="hello world")})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == "hello world"

    async def test_int_repr(self):
        """Non-string input → repr() applied."""
        stage = FormatterStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": AnyContainer(data=42)})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == "42"

    async def test_list_repr(self):
        """List input → repr() shows list notation."""
        stage = FormatterStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": AnyContainer(data=[1, 2, 3])})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == "[1, 2, 3]"

    async def test_dict_repr(self):
        """Dict input → repr() applied."""
        stage = FormatterStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": AnyContainer(data={"key": "value"})})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert "key" in result.output.data
        assert "value" in result.output.data

    async def test_custom_subclass_override(self):
        """Subclass can override format_value."""

        class UpperFormatterStage(FormatterStage):
            def format_value(self, data: Any) -> str:
                return str(data).upper()

        stage = UpperFormatterStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": AnyContainer(data="hello")})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == "HELLO"

    async def test_empty_string_not_skipped(self):
        """Empty string is not None → not skipped."""
        stage = FormatterStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": AnyContainer(data="")})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == ""

    async def test_false_not_skipped(self):
        """False is not None → not skipped."""
        stage = FormatterStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": AnyContainer(data=False)})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == "False"

    async def test_zero_not_skipped(self):
        """0 is not None → not skipped."""
        stage = FormatterStage(timeout=5.0)
        stage.__class__.cache_handler = NO_CACHE
        stage.attach_inputs({"data": AnyContainer(data=0)})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.data == "0"
