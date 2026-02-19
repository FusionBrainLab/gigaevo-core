# gigaevo/programs/stages/formatter.py
from __future__ import annotations

from typing import Any, cast

from gigaevo.programs.core_types import ProgramStageResult
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import AnyContainer, StageIO, StringContainer
from gigaevo.programs.stages.stage_registry import StageRegistry


class FormatterInputs(StageIO):
    """Single input: any value to format (AnyContainer)."""

    data: AnyContainer


@StageRegistry.register(
    description="Format any value to string; subclass and override format_value() to customize."
)
class FormatterStage(Stage):
    """
    General-purpose formatter: converts any input to a string for downstream use
    (e.g. mutation context).

    **Input**: ``AnyContainer``.
    **Output**: ``StringContainer`` — the result of :meth:`format_value`.

    Subclass and override :meth:`format_value` to implement your own logic
    (e.g. artifact summaries, array rendering, or domain-specific text).
    """

    InputsModel = FormatterInputs
    OutputModel = StringContainer

    def format_value(self, data: Any) -> str:
        """
        Turn the input value into a string. Override in subclasses for custom
        formatting (e.g. truncation, summarization, or structured output).
        Not called when data is None; compute() returns skipped instead.
        """
        if isinstance(data, str):
            return data
        return repr(data)

    async def compute(self, program: Program) -> StringContainer | ProgramStageResult:
        params = cast(FormatterInputs, self.params)
        value = params.data.data
        if value is None:
            return ProgramStageResult.skipped(
                message="No data to format",
                stage=self.__class__.__name__,
            )
        formatted = self.format_value(value)
        return StringContainer(data=formatted)
