"""Tests for STAGE_EXEC canonical-event emission from Stage.execute().

The base class is the single emission seam for STAGE_EXEC on the
miss/rerun path (cache-hit emission lives at the DAG level because `execute()`
never runs for a cache-hit).

After Stage.execute() finishes (success OR failure OR timeout), exactly one
[STAGE_EXEC] {json} line must land, with:
- stage == stage_name
- program_id == program.id
- decision in {"miss", "no_cache"}
- duration_ms >= 0
"""

from __future__ import annotations

import json
import re

from loguru import logger
import pytest

from gigaevo.programs.core_types import StageIO, StageState, VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE, InputHashCache


class _Out(StageIO):
    v: int = 1


class _OkStage(Stage):
    InputsModel = VoidInput
    OutputModel = _Out
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> _Out:
        return _Out(v=7)


class _BoomStage(Stage):
    InputsModel = VoidInput
    OutputModel = _Out
    cache_handler = NO_CACHE

    async def compute(self, program: Program) -> _Out:
        raise RuntimeError("kaboom")


class _CachedStage(Stage):
    InputsModel = VoidInput
    OutputModel = _Out
    cache_handler = InputHashCache()

    async def compute(self, program: Program) -> _Out:
        return _Out(v=3)


@pytest.fixture
def log_sink():
    captured: list[tuple[str, dict]] = []

    def sink(message):
        captured.append((str(message), dict(message.record)))

    sink_id = logger.add(sink, level="DEBUG", format="{message}")
    yield captured
    logger.remove(sink_id)


def _stage_exec_lines(log_sink):
    return [m for m, _ in log_sink if "[STAGE_EXEC]" in m]


def _prog() -> Program:
    return Program(code="def solve(): return 1", state=ProgramState.RUNNING)


class TestStageExecEmits:
    async def test_success_emits_single_stage_exec_event(self, log_sink):
        stage = _OkStage(timeout=5.0)
        stage.attach_inputs({})
        prog = _prog()
        result = await stage.execute(prog)
        assert result.status == StageState.COMPLETED

        lines = _stage_exec_lines(log_sink)
        assert len(lines) == 1, f"expected 1 STAGE_EXEC line, got {lines}"

        match = re.search(r"\[STAGE_EXEC\]\s+(\{.*\})\s*$", lines[0])
        assert match, f"line not in [EVENT] {{json}} shape: {lines[0]!r}"
        body = json.loads(match.group(1))
        assert body["event"] == "STAGE_EXEC"
        assert body["stage"] == stage.stage_name
        assert body["program_id"] == prog.id
        # NO_CACHE → decision is "no_cache" or "miss" (first run, no prior result)
        assert body["decision"] in {"no_cache", "miss"}
        assert body["duration_ms"] >= 0.0

    async def test_failure_still_emits_stage_exec_event(self, log_sink):
        stage = _BoomStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())
        assert result.status == StageState.FAILED

        lines = _stage_exec_lines(log_sink)
        assert len(lines) == 1, f"expected STAGE_EXEC even on failure, got {lines}"
        body = json.loads(re.search(r"\{.*\}$", lines[0]).group(0))
        assert body["event"] == "STAGE_EXEC"
        assert body["duration_ms"] >= 0.0

    async def test_inputhash_cached_miss_labelled_miss(self, log_sink):
        stage = _CachedStage(timeout=5.0)
        stage.attach_inputs({})
        await stage.execute(_prog())

        lines = _stage_exec_lines(log_sink)
        assert len(lines) == 1
        body = json.loads(re.search(r"\{.*\}$", lines[0]).group(0))
        # InputHashCache on first run => decision="miss" (stored hash was None)
        assert body["decision"] == "miss"
