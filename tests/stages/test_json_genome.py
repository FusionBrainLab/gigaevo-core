"""Tests for ParseJsonProgram: JSON-document genomes parse instead of executing."""

from __future__ import annotations

import pytest

from gigaevo.programs.program import Program
from gigaevo.programs.stages.json_genome import ParseJsonProgram


def _stage() -> ParseJsonProgram:
    return ParseJsonProgram(timeout=30.0)


async def test_parses_json_document_into_box():
    out = await _stage().compute(
        Program(code='{"steps": [{"number": 1}]}', iteration=0)
    )
    assert out.data == {"steps": [{"number": 1}]}


async def test_invalid_json_raises_labeled_error():
    with pytest.raises(ValueError, match="json_parse_error"):
        await _stage().compute(Program(code="{'steps': 1}", iteration=0))


async def test_empty_code_raises():
    # Program model enforces min_length=1 on code, so bypass validation with model_construct
    program = Program.model_construct(code="   ", iteration=0)
    with pytest.raises(ValueError, match="empty"):
        await _stage().compute(program)
