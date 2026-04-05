"""Shared fixtures for memory tests."""

from pathlib import Path

import pytest

from tests.fakes.agentic_memory import make_test_memory


@pytest.fixture
def make_memory(tmp_path: Path):
    """Factory fixture to create AmemGamMemory instances with sensible test defaults.

    Usage::

        mem = make_memory()                                       # defaults
        mem = make_memory(search_limit=10)                        # override
        mem = make_memory(card_update_dedup_config={"enabled": True})
    """

    def _make_memory(**overrides):
        return make_test_memory(tmp_path, **overrides)

    return _make_memory
