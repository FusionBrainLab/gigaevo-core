"""Tests for LineageStage instrumentation (baseline logging)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from loguru import logger
import pytest

from gigaevo.programs.program import Program
from gigaevo.programs.stages.common import CacheOnlyInput


@pytest.mark.asyncio
async def test_lineage_stage_logs_n_parents():
    """Base LineageStage logs program id and n_parents on every invocation."""
    from gigaevo.programs.stages.insights_lineage import LineageStage

    storage = MagicMock()
    storage.mget = AsyncMock(
        return_value=[MagicMock(spec=Program), MagicMock(spec=Program)]
    )

    stage = LineageStage.__new__(LineageStage)
    stage.storage = storage

    program = MagicMock(spec=Program)
    program.id = "abcdef1234-child"
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
