"""Tests for gigaevo.adversarial.opponent_provider."""

from __future__ import annotations

import asyncio
import json

import pytest

from gigaevo.adversarial.opponent_provider import (
    OpponentArchiveProvider,
    OpponentProgram,
    RedisOpponentArchiveProvider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeOpponentProvider(OpponentArchiveProvider):
    """In-memory provider for testing."""

    def __init__(self, programs: list[OpponentProgram] | None = None):
        self._programs = programs or []

    async def get_opponents(self, n: int = 5) -> list[OpponentProgram]:
        return self._programs[:n]


def _make_program_json(code: str, fitness: float = 0.5) -> str:
    return json.dumps({"code": code, "metrics": {"fitness": fitness}})


# ---------------------------------------------------------------------------
# Tests: FakeOpponentProvider
# ---------------------------------------------------------------------------


class TestFakeOpponentProvider:
    def test_empty_provider_returns_empty(self):
        provider = FakeOpponentProvider()
        result = asyncio.get_event_loop().run_until_complete(provider.get_opponents(5))
        assert result == []

    def test_returns_up_to_n(self):
        programs = [
            OpponentProgram(program_id=f"p{i}", code=f"code_{i}", fitness=0.5)
            for i in range(10)
        ]
        provider = FakeOpponentProvider(programs)
        result = asyncio.get_event_loop().run_until_complete(provider.get_opponents(3))
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: RedisOpponentArchiveProvider
# ---------------------------------------------------------------------------


class TestRedisOpponentArchiveProvider:
    def test_requires_sources(self):
        provider = RedisOpponentArchiveProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "test/run"}],
        )
        assert len(provider._sources) == 1
        assert provider._sources[0] == (1, "test/run")

    def test_default_island_id(self):
        provider = RedisOpponentArchiveProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "test"}],
        )
        assert provider._island_id == "fitness_island"

    def test_custom_island_id(self):
        provider = RedisOpponentArchiveProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "test"}],
            island_id="custom_island",
        )
        assert provider._island_id == "custom_island"

    def test_cache_ttl_default(self):
        provider = RedisOpponentArchiveProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "test"}],
        )
        assert provider._cache_ttl == 30.0

    def test_multiple_sources(self):
        provider = RedisOpponentArchiveProvider(
            host="localhost",
            port=6379,
            sources=[
                {"db": 1, "prefix": "run_a"},
                {"db": 2, "prefix": "run_b"},
            ],
        )
        assert len(provider._sources) == 2

    @pytest.mark.asyncio
    async def test_empty_cache_returns_empty(self):
        """get_opponents on a fresh provider with no Redis returns empty."""
        provider = RedisOpponentArchiveProvider(
            host="localhost",
            port=99999,  # intentionally wrong port
            sources=[{"db": 1, "prefix": "test"}],
        )
        result = await provider.get_opponents(5)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_opponents_with_preloaded_cache(self):
        """Verify sampling behavior when cache is pre-loaded."""
        provider = RedisOpponentArchiveProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "test"}],
        )
        # Manually populate the cache to avoid needing real Redis
        import time

        provider._cache = [
            OpponentProgram(program_id=f"p{i}", code=f"code_{i}", fitness=float(i))
            for i in range(10)
        ]
        provider._cache_time = time.monotonic()

        result = await provider.get_opponents(3)
        assert len(result) == 3
        assert all(isinstance(p, OpponentProgram) for p in result)

    @pytest.mark.asyncio
    async def test_returns_all_when_n_exceeds_cache(self):
        """When n > len(cache), return all cached opponents."""
        provider = RedisOpponentArchiveProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "test"}],
        )
        import time

        provider._cache = [
            OpponentProgram(program_id="p0", code="c0", fitness=0.5),
            OpponentProgram(program_id="p1", code="c1", fitness=0.8),
        ]
        provider._cache_time = time.monotonic()

        result = await provider.get_opponents(10)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: OpponentProgram dataclass
# ---------------------------------------------------------------------------


class TestOpponentProgram:
    def test_fields(self):
        p = OpponentProgram(program_id="abc123", code="x = 1", fitness=0.75)
        assert p.program_id == "abc123"
        assert p.code == "x = 1"
        assert p.fitness == 0.75
