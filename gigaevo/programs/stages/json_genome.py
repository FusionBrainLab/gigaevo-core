"""Stage for JSON-document genomes: parse Program.code instead of executing it."""

from __future__ import annotations

import json
from typing import Any

from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import Box
from gigaevo.programs.stages.stage_registry import StageRegistry


@StageRegistry.register(description="Parse a JSON-document genome into a dict")
class ParseJsonProgram(Stage):
    """Replaces both ValidateCodeStage (parse is the syntax gate) and
    CallProgramFunction (the parsed document is the validator payload) for
    genomes that are data, not Python."""

    InputsModel = VoidInput
    OutputModel = Box[Any]

    async def compute(self, program: Program) -> Box[Any]:
        code = program.code or ""
        if not code.strip():
            raise ValueError("Genome is empty")
        try:
            payload = json.loads(code)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"json_parse_error: line {e.lineno} col {e.colno}: {e.msg}"
            ) from e
        return self.OutputModel(data=payload)
