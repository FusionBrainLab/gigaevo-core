"""Tests for MemoryContextStage and MemoryMutationContext."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.evolution.mutation.context import MemoryMutationContext
from gigaevo.llm.agents.memory_selector import MemorySelection
from gigaevo.memory.provider import NullMemoryProvider, SelectorMemoryProvider
from gigaevo.programs.program import Program
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import StringContainer
from gigaevo.programs.stages.memory_context import MemoryContextStage


def _make_program(code: str = "def solve(): return 42") -> Program:
    return Program(code=code)


class TestMemoryMutationContext:
    def test_format_with_content(self) -> None:
        ctx = MemoryMutationContext(memory_block="1. Use caching\n2. Try BFS")
        result = ctx.format()
        assert result.startswith("## Memory Instructions")
        assert "1. Use caching" in result
        assert "2. Try BFS" in result

    def test_format_empty_returns_empty(self) -> None:
        ctx = MemoryMutationContext(memory_block="")
        assert ctx.format() == ""

    def test_format_whitespace_only_returns_empty(self) -> None:
        ctx = MemoryMutationContext(memory_block="   \n  ")
        assert ctx.format() == ""


class TestMemoryContextStageWithNullProvider:
    @pytest.mark.asyncio
    async def test_returns_empty_string(self) -> None:
        stage = MemoryContextStage(
            memory_provider=NullMemoryProvider(),
            task_description="multi-hop QA",
            metrics_description="fitness: higher is better",
            timeout=60,
        )
        program = _make_program()
        result = await stage.compute(program)
        assert isinstance(result, StringContainer)
        assert result.data == ""

    @pytest.mark.asyncio
    async def test_does_not_write_metadata(self) -> None:
        stage = MemoryContextStage(
            memory_provider=NullMemoryProvider(),
            task_description="t",
            metrics_description="m",
            timeout=60,
        )
        program = _make_program()
        await stage.compute(program)
        assert MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY not in program.metadata


class TestMemoryContextStageWithSelectorProvider:
    @pytest.mark.asyncio
    async def test_returns_formatted_cards(self) -> None:
        mock_selector = AsyncMock()
        mock_selector.select.return_value = MemorySelection(
            cards=["1. Use caching for repeated lookups", "2. Try BFS over DFS"],
            card_ids=["card-abc", "card-def"],
        )

        provider = SelectorMemoryProvider(max_cards=3)
        provider._selector = mock_selector

        stage = MemoryContextStage(
            memory_provider=provider,
            task_description="multi-hop QA",
            metrics_description="fitness",
            timeout=60,
        )
        program = _make_program()
        result = await stage.compute(program)

        assert isinstance(result, StringContainer)
        assert "1. Use caching for repeated lookups" in result.data
        assert "2. Try BFS over DFS" in result.data

    @pytest.mark.asyncio
    async def test_writes_card_ids_to_metadata(self) -> None:
        mock_selector = AsyncMock()
        mock_selector.select.return_value = MemorySelection(
            cards=["idea1"],
            card_ids=["card-abc-123"],
        )

        provider = SelectorMemoryProvider(max_cards=1)
        provider._selector = mock_selector

        stage = MemoryContextStage(
            memory_provider=provider,
            task_description="t",
            metrics_description="m",
            timeout=60,
        )
        program = _make_program()
        await stage.compute(program)

        assert program.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == [
            "card-abc-123"
        ]

    @pytest.mark.asyncio
    async def test_empty_selection_returns_empty_string(self) -> None:
        mock_selector = AsyncMock()
        mock_selector.select.return_value = MemorySelection(cards=[], card_ids=[])

        provider = SelectorMemoryProvider(max_cards=3)
        provider._selector = mock_selector

        stage = MemoryContextStage(
            memory_provider=provider,
            task_description="t",
            metrics_description="m",
            timeout=60,
        )
        program = _make_program()
        result = await stage.compute(program)

        assert result.data == ""
        assert MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY not in program.metadata


class TestMemoryContextStageProperties:
    def test_no_cache(self) -> None:
        assert MemoryContextStage.cache_handler is NO_CACHE
