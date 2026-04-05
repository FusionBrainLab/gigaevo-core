"""Shared fixtures for memory tests."""

from pathlib import Path

import pytest

from gigaevo.memory.shared_memory.memory import AmemGamMemory


@pytest.fixture
def make_memory(tmp_path: Path):
    """Factory fixture to create AmemGamMemory instances with sensible test defaults."""

    def _make_memory(**overrides):
        defaults = dict(
            checkpoint_path=str(tmp_path / "mem"),
            use_api=False,
            sync_on_init=False,
            enable_llm_synthesis=False,
            enable_memory_evolution=False,
            enable_llm_card_enrichment=False,
        )
        defaults.update(overrides)
        return AmemGamMemory(**defaults)

    return _make_memory
