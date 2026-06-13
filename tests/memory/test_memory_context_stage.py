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
from gigaevo.memory.core import AuctionBid, MemorySelection
from gigaevo.memory.provider import (
    MemoryProvider,
    NullMemoryProvider,
    SelectorMemoryProvider,
)
from gigaevo.programs.program import Program
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import StringContainer
from gigaevo.programs.stages.memory_context import (
    MemoryContextStage,
    MemoryExposureCounter,
)


def _make_program(code: str = "def solve(): return 42") -> Program:
    return Program(code=code)


def _bid(
    card_id: str, posterior_a: float, posterior_b: float, selected: bool
) -> AuctionBid:
    return AuctionBid(
        card_id=card_id,
        posterior_a=posterior_a,
        posterior_b=posterior_b,
        theta=0.5,
        baseline_a=3.0,
        baseline_b=3.0,
        baseline_theta=0.4,
        selected=selected,
    )


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
    async def test_writes_empty_selection_metadata(self) -> None:
        # Empty must be written explicitly so a child's birth stamp can
        # distinguish "no cards at birth" from legacy programs with no record.
        stage = MemoryContextStage(
            memory_provider=NullMemoryProvider(),
            task_description="t",
            metrics_description="m",
            timeout=60,
        )
        program = _make_program()
        await stage.compute(program)
        assert program.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == []
        assert program.metadata[MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY] == []


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
        assert program.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == []

    @pytest.mark.asyncio
    async def test_writes_candidate_slate_to_metadata(self) -> None:
        slate = [
            _bid("card-abc", 200.0, 1.0, selected=True),
            _bid("card-xyz", 1.0, 200.0, selected=False),
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

        assert program.metadata[MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY] == [
            bid.model_dump() for bid in slate
        ]

    @pytest.mark.asyncio
    async def test_writes_slate_even_when_auction_selects_nothing(self) -> None:
        # The "no-card" outcome is the whole point of the auction: a 0-winner
        # sweep must still record which candidates were offered and rejected.
        slate = [_bid("card-xyz", 1.0, 200.0, selected=False)]
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
        assert program.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == []
        assert program.metadata[MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY] == [
            bid.model_dump() for bid in slate
        ]

    @pytest.mark.asyncio
    async def test_empty_slate_writes_empty_list(self) -> None:
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

        assert program.metadata[MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY] == []

    @pytest.mark.asyncio
    async def test_stale_metadata_cleared_when_selection_raises(self) -> None:
        # A requeued parent re-runs this NO_CACHE stage; if select_cards raises
        # (e.g. a stage timeout), the previous run's slate must already be
        # erased, or the failed sweep leaves children inheriting phantom credit.
        class _RaisingProvider(MemoryProvider):
            async def select_cards(
                self,
                program: Program,
                *,
                task_description: str,
                metrics_description: str,
            ) -> MemorySelection:
                raise RuntimeError("select timed out")

        stage = MemoryContextStage(
            memory_provider=_RaisingProvider(),
            task_description="t",
            metrics_description="m",
            timeout=60,
        )
        program = _make_program()
        program.set_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, ["stale-card"])
        program.set_metadata(
            MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY,
            [_bid("stale-card", 5.0, 1.0, selected=True).model_dump()],
        )

        with pytest.raises(RuntimeError):
            await stage.compute(program)

        assert program.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == []
        assert program.metadata[MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY] == []

    @pytest.mark.asyncio
    async def test_stale_metadata_overwritten_by_empty_selection(self) -> None:
        # A requeued parent re-runs this NO_CACHE stage; a now-empty auction
        # must erase the previous slate, or children inherit phantom credit.
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
        program.set_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, ["stale-card"])
        program.set_metadata(
            MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY,
            [_bid("stale-card", 5.0, 1.0, selected=True).model_dump()],
        )
        await stage.compute(program)

        assert program.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == []
        assert program.metadata[MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY] == []


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


class TestExposureCounter:
    def _stage(
        self, selection: MemorySelection, exposure: MemoryExposureCounter
    ) -> MemoryContextStage:
        mock_selector = AsyncMock()
        mock_selector.select.return_value = selection
        provider = SelectorMemoryProvider(
            backend=LocalMemoryBackendFactory(), max_cards=3
        )
        provider._pipeline = mock_selector
        return MemoryContextStage(
            memory_provider=provider,
            task_description="t",
            metrics_description="m",
            timeout=60,
            exposure=exposure,
        )

    @pytest.mark.asyncio
    async def test_counts_shared_across_stage_instances(self) -> None:
        # The DAG builds a fresh stage per program; the counter is the
        # run-lifetime object shared through the pipeline-builder closure.
        exposure = MemoryExposureCounter()
        non_empty = self._stage(
            MemorySelection(cards=["idea"], card_ids=["card-a"]), exposure
        )
        empty = self._stage(MemorySelection(cards=[], card_ids=[]), exposure)

        await non_empty.compute(_make_program())
        await empty.compute(_make_program())
        await empty.compute(_make_program())

        assert exposure.attempts == 3
        assert exposure.non_empty == 1

    @pytest.mark.asyncio
    async def test_null_provider_counts_as_empty_attempt(self) -> None:
        exposure = MemoryExposureCounter()
        stage = MemoryContextStage(
            memory_provider=NullMemoryProvider(),
            task_description="t",
            metrics_description="m",
            timeout=60,
            exposure=exposure,
        )
        await stage.compute(_make_program())
        assert exposure.attempts == 1
        assert exposure.non_empty == 0
