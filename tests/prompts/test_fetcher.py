"""Tests for PromptFetcher abstraction and implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from gigaevo.llm.bandit import MutationOutcome
from gigaevo.prompts.fetcher import (
    FetchedPrompt,
    FixedDirPromptFetcher,
    GigaEvoArchivePromptFetcher,
    PromptFetcher,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_prompts_dir(tmp_path: Path) -> Path:
    """Create a temporary prompts directory with test fixtures."""
    # Create mutation/system.txt and mutation/user.txt
    mutation_dir = tmp_path / "mutation"
    mutation_dir.mkdir()
    (mutation_dir / "system.txt").write_text(
        "System prompt for {task_description}: {metrics_description}"
    )
    (mutation_dir / "user.txt").write_text("User prompt for {code}")

    # Create insights/system.txt and insights/user.txt
    insights_dir = tmp_path / "insights"
    insights_dir.mkdir()
    (insights_dir / "system.txt").write_text("Insights system")
    (insights_dir / "user.txt").write_text("Insights user")

    return tmp_path


# ---------------------------------------------------------------------------
# FixedDirPromptFetcher Tests
# ---------------------------------------------------------------------------


class TestFixedDirPromptFetcher:
    """Tests for FixedDirPromptFetcher."""

    def test_initialization(self):
        """FixedDirPromptFetcher can be initialized."""
        fetcher = FixedDirPromptFetcher(prompts_dir=None)
        assert fetcher is not None
        assert fetcher.is_dynamic is False

    def test_fetch_from_custom_dir(self, tmp_prompts_dir: Path):
        """fetch() reads from custom prompts directory."""
        fetcher = FixedDirPromptFetcher(prompts_dir=tmp_prompts_dir)
        result = fetcher.fetch("mutation", "system")
        assert (
            result.text == "System prompt for {task_description}: {metrics_description}"
        )
        assert result.prompt_id is None

    def test_fetch_fallback_to_package_defaults(self):
        """fetch() falls back to package defaults when file missing."""
        # This will use package defaults (gigaevo/prompts/mutation/system.txt)
        fetcher = FixedDirPromptFetcher(prompts_dir=None)
        result = fetcher.fetch("mutation", "system")
        assert result.text is not None
        assert len(result.text) > 0
        assert result.prompt_id is None

    def test_fetch_caches_results(self, tmp_prompts_dir: Path):
        """fetch() caches results on repeated calls."""
        fetcher = FixedDirPromptFetcher(prompts_dir=tmp_prompts_dir)
        result1 = fetcher.fetch("mutation", "system")
        result2 = fetcher.fetch("mutation", "system")
        assert result1 is result2  # Same object

    def test_different_prompt_types(self, tmp_prompts_dir: Path):
        """fetch() returns different content for system vs user."""
        fetcher = FixedDirPromptFetcher(prompts_dir=tmp_prompts_dir)
        system = fetcher.fetch("mutation", "system")
        user = fetcher.fetch("mutation", "user")
        assert system.text != user.text

    def test_get_stats_returns_empty_dict(self, tmp_prompts_dir: Path):
        """get_stats() returns empty dict for FixedDirPromptFetcher."""
        fetcher = FixedDirPromptFetcher(prompts_dir=tmp_prompts_dir)
        stats = fetcher.get_stats()
        assert stats == {}


# ---------------------------------------------------------------------------
# GigaEvoArchivePromptFetcher Tests
# ---------------------------------------------------------------------------


class TestGigaEvoArchivePromptFetcher:
    """Tests for GigaEvoArchivePromptFetcher (with mocks)."""

    def test_initialization(self, tmp_prompts_dir: Path):
        """GigaEvoArchivePromptFetcher can be initialized."""
        fetcher = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="chains/hotpotqa",
            fallback_prompts_dir=tmp_prompts_dir,
        )
        assert fetcher is not None
        assert fetcher.is_dynamic is True

    def test_fetch_returns_fallback_initially(self, tmp_prompts_dir: Path):
        """fetch() returns fallback when no champion available."""
        fetcher = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="chains/hotpotqa",
            fallback_prompts_dir=tmp_prompts_dir,
        )
        # No Redis available, should fall back
        result = fetcher.fetch("mutation", "system")
        assert (
            result.text == "System prompt for {task_description}: {metrics_description}"
        )
        assert result.prompt_id is None

    def test_record_outcome_skips_rejected_acceptor(self, tmp_prompts_dir: Path):
        """record_outcome() skips REJECTED_ACCEPTOR outcomes."""
        fetcher = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="chains/hotpotqa",
            fallback_prompts_dir=tmp_prompts_dir,
        )
        # Should not raise when prompt_id is None or outcome is REJECTED_ACCEPTOR
        fetcher.record_outcome(
            prompt_id="abc123",
            child_fitness=0.5,
            parent_fitness=0.4,
            higher_is_better=True,
            outcome=MutationOutcome.REJECTED_ACCEPTOR,
        )
        # No assertion needed — just verify no exception

    def test_record_outcome_noop_when_prompt_id_none(self, tmp_prompts_dir: Path):
        """record_outcome() is no-op when prompt_id is None."""
        fetcher = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="chains/hotpotqa",
            fallback_prompts_dir=tmp_prompts_dir,
        )
        fetcher.record_outcome(
            prompt_id=None,
            child_fitness=0.5,
            parent_fitness=0.4,
            higher_is_better=True,
            outcome=MutationOutcome.ACCEPTED,
        )
        # No assertion needed — just verify no exception

    def test_get_stats_includes_cache_info(self, tmp_prompts_dir: Path):
        """get_stats() returns cache hit/error counts."""
        fetcher = GigaEvoArchivePromptFetcher(
            prompt_redis_db=6,
            main_redis_prefix="chains/hotpotqa",
            fallback_prompts_dir=tmp_prompts_dir,
        )
        stats = fetcher.get_stats()
        assert "cache_hits" in stats
        assert "fetch_errors" in stats
        assert "has_champion" in stats


# ---------------------------------------------------------------------------
# FetchedPrompt Tests
# ---------------------------------------------------------------------------


class TestFetchedPrompt:
    """Tests for FetchedPrompt dataclass."""

    def test_creation_with_id(self):
        """FetchedPrompt can be created with prompt_id."""
        prompt = FetchedPrompt(text="Hello", prompt_id="abc123")
        assert prompt.text == "Hello"
        assert prompt.prompt_id == "abc123"

    def test_creation_without_id(self):
        """FetchedPrompt can be created with prompt_id=None."""
        prompt = FetchedPrompt(text="Hello", prompt_id=None)
        assert prompt.text == "Hello"
        assert prompt.prompt_id is None


# ---------------------------------------------------------------------------
# PromptFetcher ABC Tests
# ---------------------------------------------------------------------------


class TestPromptFetcherABC:
    """Tests for PromptFetcher abstract base class."""

    def test_cannot_instantiate_directly(self):
        """PromptFetcher cannot be instantiated (abstract)."""
        with pytest.raises(TypeError):
            PromptFetcher()  # type: ignore

    def test_subclass_must_implement_fetch(self):
        """PromptFetcher subclasses must implement fetch()."""

        class IncompleteFetcher(PromptFetcher):
            pass

        with pytest.raises(TypeError):
            IncompleteFetcher()  # type: ignore

    def test_record_outcome_default_noop(self):
        """PromptFetcher.record_outcome() default is no-op."""
        fetcher = FixedDirPromptFetcher()
        # Should not raise
        fetcher.record_outcome(
            prompt_id="test",
            child_fitness=0.5,
            parent_fitness=0.4,
            higher_is_better=True,
            outcome=MutationOutcome.ACCEPTED,
        )
