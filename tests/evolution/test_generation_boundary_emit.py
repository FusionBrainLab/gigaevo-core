"""Tests for GENERATION_BOUNDARY canonical-event emission.

The engine tick (line where `total_generations` bumps) is the single seam
for GENERATION_BOUNDARY. Every generation must emit exactly one
`[GENERATION_BOUNDARY] {json}` line with the post-increment `gen`.
"""

from __future__ import annotations

import asyncio
import json
import re
from unittest.mock import AsyncMock, MagicMock

from loguru import logger
import pytest

from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine

ENGINE_TEST_TIMEOUT = 5.0


def _make_engine() -> EvolutionEngine:
    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()

    storage.count_by_status.return_value = 0
    storage.get_all_by_status.return_value = []
    storage.get_ids_by_status.return_value = []
    storage.mget.return_value = []
    storage.snapshot = MagicMock()
    strategy.select_elites.return_value = []
    strategy.get_program_ids.return_value = []

    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=EngineConfig(),
        writer=writer,
        metrics_tracker=metrics_tracker,
    )
    engine.state = AsyncMock()
    return engine


@pytest.fixture
def log_sink():
    captured: list[str] = []

    def sink(message):
        captured.append(str(message))

    sink_id = logger.add(sink, level="DEBUG", format="{message}")
    yield captured
    logger.remove(sink_id)


def _gb_lines(captured):
    return [m for m in captured if "[GENERATION_BOUNDARY]" in m]


class TestGenerationBoundaryEmits:
    async def test_step_emits_single_generation_boundary(self, log_sink):
        engine = _make_engine()

        await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

        lines = _gb_lines(log_sink)
        assert len(lines) == 1, f"expected exactly one GENERATION_BOUNDARY, got {lines}"

    async def test_generation_boundary_carries_post_tick_gen(self, log_sink):
        engine = _make_engine()
        engine.metrics.total_generations = 4

        await asyncio.wait_for(engine.step(), timeout=ENGINE_TEST_TIMEOUT)

        lines = _gb_lines(log_sink)
        assert len(lines) == 1
        body = json.loads(re.search(r"\{.*\}$", lines[0]).group(0))
        assert body["event"] == "GENERATION_BOUNDARY"
        # The event fires after total_generations ticks — so gen == 5 here.
        assert body["gen"] == 5
