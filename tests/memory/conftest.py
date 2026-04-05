"""Shared fixtures for memory tests."""

from pathlib import Path

import pytest

from gigaevo.memory.shared_memory.memory import AmemGamMemory


@pytest.fixture
def make_memory(tmp_path: Path):
    """Factory fixture to create AmemGamMemory instances with sensible test defaults.

    Supports both legacy kwargs and the new MemoryConfig API::

        mem = make_memory()                          # defaults
        mem = make_memory(search_limit=10)           # legacy kwarg override
        mem = make_memory(config=MemoryConfig(...))  # new API
    """

    def _make_memory(**overrides):
        if "config" in overrides:
            return AmemGamMemory(config=overrides.pop("config"), **overrides)
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
