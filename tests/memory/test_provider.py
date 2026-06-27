"""Tests for gigaevo.memory.provider — MemoryProvider abstraction."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.backend_factory import LocalMemoryBackendFactory
from gigaevo.memory.core import (
    GamRetriever,
    MemoryReadPipeline,
    MemorySelection,
    ThompsonAuctioneer,
)
from gigaevo.memory.provider import (
    MemoryProvider,
    NullMemoryProvider,
    SelectorMemoryProvider,
)
from gigaevo.memory.shared_memory.memory_config import GamConfig
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
    async def test_delegates_to_pipeline(self) -> None:
        mock_pipeline = AsyncMock()
        expected = MemorySelection(
            cards=["1. Use caching for repeated lookups"],
            card_ids=["card-abc-123"],
        )
        mock_pipeline.select.return_value = expected

        provider = SelectorMemoryProvider(
            backend=LocalMemoryBackendFactory(), max_cards=3
        )
        provider._pipeline = mock_pipeline

        program = _make_program()
        result = await provider.select_cards(
            program=program,
            task_description="multi-hop QA",
            metrics_description="fitness: fraction correct",
        )

        assert result is expected
        mock_pipeline.select.assert_called_once()
        call_kwargs = mock_pipeline.select.call_args.kwargs
        assert call_kwargs["parents"] == [program]
        assert call_kwargs["task_description"] == "multi-hop QA"
        assert call_kwargs["metrics_description"] == "fitness: fraction correct"
        assert call_kwargs["max_cards"] == 3

    @pytest.mark.asyncio
    async def test_passes_max_cards(self) -> None:
        mock_pipeline = AsyncMock()
        mock_pipeline.select.return_value = MemorySelection(cards=[], card_ids=[])

        provider = SelectorMemoryProvider(
            backend=LocalMemoryBackendFactory(), max_cards=7
        )
        provider._pipeline = mock_pipeline

        await provider.select_cards(
            program=_make_program(),
            task_description="t",
            metrics_description="m",
        )

        assert mock_pipeline.select.call_args.kwargs["max_cards"] == 7

    @pytest.mark.asyncio
    async def test_default_max_cards_matches_shipped_config(self) -> None:
        """config/memory/local.yaml ships max_cards=1; the constructor default
        must agree so a bare provider behaves like a configured run."""
        mock_pipeline = AsyncMock()
        mock_pipeline.select.return_value = MemorySelection(cards=[], card_ids=[])

        provider = SelectorMemoryProvider(backend=LocalMemoryBackendFactory())
        provider._pipeline = mock_pipeline

        await provider.select_cards(
            program=_make_program(),
            task_description="t",
            metrics_description="m",
        )

        assert mock_pipeline.select.call_args.kwargs["max_cards"] == 1

    def test_bare_retriever_defaults_match_shipped_gam_yaml(self) -> None:
        """A directly-constructed GamRetriever (no Hydra) must compose the same
        retrieval as config/memory/retriever/gam.yaml: page_index+vector only
        and the shipped max_iters=3 — so tests and scripts behave like a real
        run."""
        retriever = GamRetriever()
        assert retriever.allowed_tools == ["page_index", "vector"]
        assert retriever.max_iters == 3

    @pytest.mark.asyncio
    async def test_passes_mutation_mode_rewrite(self) -> None:
        mock_pipeline = AsyncMock()
        mock_pipeline.select.return_value = MemorySelection(cards=[], card_ids=[])

        provider = SelectorMemoryProvider(
            backend=LocalMemoryBackendFactory(), max_cards=1
        )
        provider._pipeline = mock_pipeline

        await provider.select_cards(
            program=_make_program(),
            task_description="t",
            metrics_description="m",
        )

        assert mock_pipeline.select.call_args.kwargs["mutation_mode"] == "rewrite"

    @pytest.mark.asyncio
    async def test_backend_built_lazily_and_reused(self) -> None:
        with patch.object(
            LocalMemoryBackendFactory, "build", return_value=MagicMock()
        ) as mock_build:
            provider = SelectorMemoryProvider(
                backend=LocalMemoryBackendFactory(), max_cards=3
            )
            mock_build.assert_not_called()

            await provider.select_cards(
                program=_make_program(),
                task_description="t",
                metrics_description="m",
            )
            mock_build.assert_called_once()

            await provider.select_cards(
                program=_make_program(),
                task_description="t2",
                metrics_description="m2",
            )
            mock_build.assert_called_once()

    @pytest.mark.asyncio
    async def test_backend_build_runs_off_event_loop(self) -> None:
        # The backend build loads an embedding model — seconds of blocking work
        # that must not stall the asyncio loop the other program-stages share.
        import threading

        main_thread = threading.current_thread()
        captured: dict[str, threading.Thread] = {}

        def _capture_build(**kwargs: object) -> MagicMock:
            captured["thread"] = threading.current_thread()
            return MagicMock()

        with patch.object(
            LocalMemoryBackendFactory, "build", side_effect=_capture_build
        ):
            provider = SelectorMemoryProvider(
                backend=LocalMemoryBackendFactory(), max_cards=1
            )
            await provider.select_cards(
                program=_make_program(),
                task_description="t",
                metrics_description="m",
            )

        assert captured["thread"] is not main_thread

    @pytest.mark.asyncio
    async def test_concurrent_select_builds_backend_once(self) -> None:
        # Two parents entering select_cards before the pipeline exists must not
        # each spawn a backend build; the once-lock serializes the off-loop
        # build that the previous test pushed into a worker thread.
        with patch.object(
            LocalMemoryBackendFactory, "build", return_value=MagicMock()
        ) as mock_build:
            provider = SelectorMemoryProvider(
                backend=LocalMemoryBackendFactory(), max_cards=1
            )
            await asyncio.gather(
                *[
                    provider.select_cards(
                        program=_make_program(),
                        task_description="t",
                        metrics_description="m",
                    )
                    for _ in range(4)
                ]
            )
            mock_build.assert_called_once()

    @pytest.mark.asyncio
    async def test_backend_failure_propagates(self) -> None:
        # Fail-fast contract: a misconfigured backend aborts the run instead of
        # silently degrading to a no-memory run.
        with patch.object(
            LocalMemoryBackendFactory,
            "build",
            side_effect=MemoryStorageError("backend init failed"),
        ):
            provider = SelectorMemoryProvider(
                backend=LocalMemoryBackendFactory(), max_cards=3
            )
            with pytest.raises(MemoryStorageError):
                await provider.select_cards(
                    program=_make_program(),
                    task_description="t",
                    metrics_description="m",
                )

    def test_checkpoint_dir_flows_to_backend_factory(self) -> None:
        with patch.object(
            LocalMemoryBackendFactory, "build", return_value=MagicMock()
        ) as mock_build:
            provider = SelectorMemoryProvider(
                backend=LocalMemoryBackendFactory(),
                max_cards=3,
                checkpoint_dir="/data/memory",
            )
            provider._get_pipeline()
            assert mock_build.call_args.kwargs["checkpoint_dir"] == "/data/memory"

    def test_injected_backend_factory_is_used(self) -> None:
        factory = LocalMemoryBackendFactory()
        with patch.object(
            LocalMemoryBackendFactory, "build", return_value=MagicMock()
        ) as mock_build:
            provider = SelectorMemoryProvider(max_cards=3, backend=factory)
            provider._get_pipeline()
            mock_build.assert_called_once()
        assert provider._backend_factory is factory

    def test_read_backend_receives_no_write_components(self) -> None:
        """The read-side backend never ingests; the evictor is a write-path
        component plumbed via IdeaTracker, not the provider."""
        with patch.object(
            LocalMemoryBackendFactory, "build", return_value=MagicMock()
        ) as mock_build:
            provider = SelectorMemoryProvider(
                backend=LocalMemoryBackendFactory(),
                max_cards=3,
            )
            provider._get_pipeline()
            assert "evictor" not in mock_build.call_args.kwargs

    def test_unbound_retriever_research_raises(self) -> None:
        with pytest.raises(RuntimeError, match="bind"):
            GamRetriever().research("query")

    def test_injected_retriever_knobs_flow_into_gam_config(self) -> None:
        retriever = GamRetriever(
            allowed_tools=["vector"],
            top_k_by_tool={"vector": 7},
            max_iters=5,
        )
        with patch.object(
            LocalMemoryBackendFactory, "build", return_value=MagicMock()
        ) as mock_build:
            provider = SelectorMemoryProvider(
                backend=LocalMemoryBackendFactory(),
                max_cards=1,
                checkpoint_dir="/data/memory",
                retriever=retriever,
            )
            provider._get_pipeline()
            assert mock_build.call_args.kwargs["gam"] == GamConfig(
                allowed_tools=["vector"],
                top_k_by_tool={"vector": 7},
                max_iters=5,
                max_cards=1,
            )

    def test_prebound_retriever_skips_backend_build(self) -> None:
        retriever = GamRetriever(backend=MagicMock())
        with patch.object(
            LocalMemoryBackendFactory, "build", return_value=MagicMock()
        ) as mock_build:
            provider = SelectorMemoryProvider(
                backend=LocalMemoryBackendFactory(), max_cards=1, retriever=retriever
            )
            provider._get_pipeline()
            mock_build.assert_not_called()

    def test_pipeline_uses_injected_components(self) -> None:
        auctioneer = ThompsonAuctioneer(baseline_prior=(5.0, 2.0))
        with patch.object(LocalMemoryBackendFactory, "build", return_value=MagicMock()):
            provider = SelectorMemoryProvider(
                backend=LocalMemoryBackendFactory(), max_cards=1, auctioneer=auctioneer
            )
            pipeline = provider._get_pipeline()
        assert isinstance(pipeline, MemoryReadPipeline)
        assert pipeline._auctioneer is auctioneer


class TestMemoryProviderIsABC:
    def test_cannot_instantiate_base(self) -> None:
        with pytest.raises(TypeError):
            MemoryProvider()  # type: ignore[abstract]


def test_selector_memory_provider_select_cards_returns_selection():
    """select_cards lazy-assembles the pipeline and returns a MemorySelection."""
    with patch.object(LocalMemoryBackendFactory, "build", return_value=MagicMock()):
        provider = SelectorMemoryProvider(
            backend=LocalMemoryBackendFactory(), max_cards=3
        )
        prog = _make_program()
        result = asyncio.run(
            provider.select_cards(
                prog,
                task_description="test task",
                metrics_description="fitness",
            )
        )

    assert isinstance(result, MemorySelection)
