"""Tests for gigaevo.memory.provider — MemoryProvider abstraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from gigaevo.llm.agents.memory_selector import MemorySelection
from gigaevo.memory.provider import (
    MemoryProvider,
    NullMemoryProvider,
    SelectorMemoryProvider,
)
from gigaevo.programs.program import Program


def _make_program(code: str = "def solve(): return 42") -> Program:
    return Program(code=code)


class TestNullMemoryProvider:
    @pytest.mark.asyncio
    async def test_returns_empty_selection(self) -> None:
        provider = NullMemoryProvider()
        result = await provider.select_cards(
            program=_make_program(),
            task_description="some task",
            metrics_description="fitness: higher is better",
        )
        assert result.cards == []
        assert result.card_ids == []

    @pytest.mark.asyncio
    async def test_returns_memory_selection_type(self) -> None:
        provider = NullMemoryProvider()
        result = await provider.select_cards(
            program=_make_program(),
            task_description="",
            metrics_description="",
        )
        assert isinstance(result, MemorySelection)


class TestSelectorMemoryProvider:
    @pytest.mark.asyncio
    async def test_delegates_to_selector_agent(self) -> None:
        mock_selector = AsyncMock()
        expected = MemorySelection(
            cards=["1. Use caching for repeated lookups"],
            card_ids=["card-abc-123"],
        )
        mock_selector.select.return_value = expected

        provider = SelectorMemoryProvider(max_cards=3)
        provider._selector = mock_selector

        program = _make_program()
        result = await provider.select_cards(
            program=program,
            task_description="multi-hop QA",
            metrics_description="fitness: fraction correct",
        )

        assert result is expected
        mock_selector.select.assert_called_once()
        call_kwargs = mock_selector.select.call_args.kwargs
        assert call_kwargs["input"] == [program]
        assert call_kwargs["task_description"] == "multi-hop QA"
        assert call_kwargs["metrics_description"] == "fitness: fraction correct"
        assert call_kwargs["max_cards"] == 3

    @pytest.mark.asyncio
    async def test_passes_max_cards(self) -> None:
        mock_selector = AsyncMock()
        mock_selector.select.return_value = MemorySelection(cards=[], card_ids=[])

        provider = SelectorMemoryProvider(max_cards=7)
        provider._selector = mock_selector

        await provider.select_cards(
            program=_make_program(),
            task_description="t",
            metrics_description="m",
        )

        assert mock_selector.select.call_args.kwargs["max_cards"] == 7

    @pytest.mark.asyncio
    async def test_passes_mutation_mode_rewrite(self) -> None:
        mock_selector = AsyncMock()
        mock_selector.select.return_value = MemorySelection(cards=[], card_ids=[])

        provider = SelectorMemoryProvider(max_cards=1)
        provider._selector = mock_selector

        await provider.select_cards(
            program=_make_program(),
            task_description="t",
            metrics_description="m",
        )

        assert mock_selector.select.call_args.kwargs["mutation_mode"] == "rewrite"

    @pytest.mark.asyncio
    async def test_init_creates_selector_lazily(self) -> None:
        """SelectorMemoryProvider defers MemorySelectorAgent creation to first use."""
        with patch(
            "gigaevo.llm.agents.memory_selector.MemorySelectorAgent"
        ) as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.select.return_value = MemorySelection(cards=[], card_ids=[])
            mock_cls.return_value = mock_instance

            provider = SelectorMemoryProvider(max_cards=3)
            # Not created yet at construction
            mock_cls.assert_not_called()

            await provider.select_cards(
                program=_make_program(),
                task_description="t",
                metrics_description="m",
            )
            # Created on first use
            mock_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_selector_reused_across_calls(self) -> None:
        """Once created, the same selector instance is reused."""
        with patch(
            "gigaevo.llm.agents.memory_selector.MemorySelectorAgent"
        ) as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.select.return_value = MemorySelection(cards=[], card_ids=[])
            mock_cls.return_value = mock_instance

            provider = SelectorMemoryProvider(max_cards=1)
            await provider.select_cards(
                program=_make_program(), task_description="t", metrics_description="m"
            )
            await provider.select_cards(
                program=_make_program(), task_description="t2", metrics_description="m2"
            )
            # Only one instance created
            mock_cls.assert_called_once()
            assert mock_instance.select.call_count == 2


class TestMemoryProviderIsABC:
    def test_cannot_instantiate_base(self) -> None:
        with pytest.raises(TypeError):
            MemoryProvider()  # type: ignore[abstract]
