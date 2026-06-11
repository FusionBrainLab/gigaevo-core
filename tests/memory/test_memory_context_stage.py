"""Tests for MemoryContextStage and MemoryMutationContext."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.evolution.mutation.context import MemoryMutationContext
from gigaevo.memory.backend_factory import LocalMemoryBackendFactory
from gigaevo.memory.core import MemorySelection
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
        assert result == "1. Use caching\n2. Try BFS"

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

        provider = SelectorMemoryProvider(
            backend=LocalMemoryBackendFactory(), max_cards=3
        )
        provider._pipeline = mock_selector

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

        provider = SelectorMemoryProvider(
            backend=LocalMemoryBackendFactory(), max_cards=1
        )
        provider._pipeline = mock_selector

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

        provider = SelectorMemoryProvider(
            backend=LocalMemoryBackendFactory(), max_cards=3
        )
        provider._pipeline = mock_selector

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

    @pytest.mark.asyncio
    async def test_writes_candidate_slate_to_metadata(self) -> None:
        slate = [
            {"card_id": "card-abc", "a": 200.0, "b": 1.0, "selected": True},
            {"card_id": "card-xyz", "a": 1.0, "b": 200.0, "selected": False},
        ]
        mock_selector = AsyncMock()
        mock_selector.select.return_value = MemorySelection(
            cards=["idea1"], card_ids=["card-abc"], slate=slate
        )

        provider = SelectorMemoryProvider(
            backend=LocalMemoryBackendFactory(), max_cards=3
        )
        provider._pipeline = mock_selector

        stage = MemoryContextStage(
            memory_provider=provider,
            task_description="t",
            metrics_description="m",
            timeout=60,
        )
        program = _make_program()
        await stage.compute(program)

        assert program.metadata[MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY] == slate

    @pytest.mark.asyncio
    async def test_writes_slate_even_when_auction_selects_nothing(self) -> None:
        # The "no-card" outcome is the whole point of the auction: a 0-winner
        # sweep must still record which candidates were offered and rejected.
        slate = [{"card_id": "card-xyz", "a": 1.0, "b": 200.0, "selected": False}]
        mock_selector = AsyncMock()
        mock_selector.select.return_value = MemorySelection(
            cards=[], card_ids=[], slate=slate
        )

        provider = SelectorMemoryProvider(
            backend=LocalMemoryBackendFactory(), max_cards=3
        )
        provider._pipeline = mock_selector

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
        assert program.metadata[MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY] == slate

    @pytest.mark.asyncio
    async def test_no_slate_metadata_when_slate_empty(self) -> None:
        mock_selector = AsyncMock()
        mock_selector.select.return_value = MemorySelection(cards=[], card_ids=[])

        provider = SelectorMemoryProvider(
            backend=LocalMemoryBackendFactory(), max_cards=3
        )
        provider._pipeline = mock_selector

        stage = MemoryContextStage(
            memory_provider=provider,
            task_description="t",
            metrics_description="m",
            timeout=60,
        )
        program = _make_program()
        await stage.compute(program)

        assert MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY not in program.metadata


class TestMemoryContextStageCardNumbering:
    @pytest.mark.asyncio
    async def test_cards_rendered_with_numbered_id_headers(self) -> None:
        mock_selector = AsyncMock()
        mock_selector.select.return_value = MemorySelection(
            cards=["lever A description", "lever B description"],
            card_ids=["card-abc", "card-def"],
        )
        provider = SelectorMemoryProvider(
            backend=LocalMemoryBackendFactory(), max_cards=3
        )
        provider._pipeline = mock_selector

        stage = MemoryContextStage(
            memory_provider=provider,
            task_description="t",
            metrics_description="m",
            timeout=60,
        )
        result = await stage.compute(_make_program())

        assert "[card 1] id=card-abc\nlever A description" in result.data
        assert "[card 2] id=card-def\nlever B description" in result.data


class TestMemoryContextStageProperties:
    def test_no_cache(self) -> None:
        assert MemoryContextStage.cache_handler is NO_CACHE
