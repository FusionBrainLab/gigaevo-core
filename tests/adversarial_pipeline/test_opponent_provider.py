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

    async def get_top_k(
        self, k: int, *, higher_is_better: bool = True
    ) -> list[OpponentProgram]:
        return sorted(
            self._programs, key=lambda o: o.fitness, reverse=higher_is_better
        )[:k]

    async def get_programs_by_ids(self, ids: list[str]) -> list[OpponentProgram]:
        id_set = set(ids)
        return [p for p in self._programs if p.program_id in id_set]

    async def get_codes_by_ids(self, ids: list[str]) -> list[str]:
        id_map = {p.program_id: p.code for p in self._programs}
        return [id_map[i] for i in ids if i in id_map]


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

    @pytest.mark.asyncio
    async def test_get_codes_by_ids_returns_matching_codes(self):
        """get_codes_by_ids serves from in-memory cache without Redis I/O."""
        import time

        provider = RedisOpponentArchiveProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "test"}],
        )
        provider._cache = [
            OpponentProgram(program_id="p0", code="code_p0", fitness=0.5),
            OpponentProgram(program_id="p1", code="code_p1", fitness=0.8),
            OpponentProgram(program_id="p2", code="code_p2", fitness=0.3),
        ]
        provider._cache_time = time.monotonic()

        codes = await provider.get_codes_by_ids(["p0", "p2"])
        assert set(codes) == {"code_p0", "code_p2"}

    @pytest.mark.asyncio
    async def test_get_codes_by_ids_skips_unknown_ids(self):
        """IDs not in cache are silently skipped."""
        import time

        provider = RedisOpponentArchiveProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "test"}],
        )
        provider._cache = [
            OpponentProgram(program_id="p0", code="code_p0", fitness=0.5),
        ]
        provider._cache_time = time.monotonic()

        codes = await provider.get_codes_by_ids(["p0", "does_not_exist"])
        assert codes == ["code_p0"]

    @pytest.mark.asyncio
    async def test_get_codes_by_ids_empty_ids_returns_empty(self):
        """Empty IDs list returns empty codes list."""
        import time

        provider = RedisOpponentArchiveProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "test"}],
        )
        provider._cache = [
            OpponentProgram(program_id="p0", code="code_p0", fitness=0.5),
        ]
        provider._cache_time = time.monotonic()

        codes = await provider.get_codes_by_ids([])
        assert codes == []


# ---------------------------------------------------------------------------
# Tests: _softmax_weights
# ---------------------------------------------------------------------------


class TestSoftmaxWeights:
    def test_identical_fitnesses_returns_uniform(self):
        from gigaevo.adversarial.opponent_provider import _softmax_weights

        weights = _softmax_weights([0.5, 0.5, 0.5])
        assert len(weights) == 3
        assert abs(weights[0] - 1 / 3) < 1e-6

    def test_weights_sum_to_one(self):
        from gigaevo.adversarial.opponent_provider import _softmax_weights

        weights = _softmax_weights([0.1, 0.5, 0.9])
        assert abs(sum(weights) - 1.0) < 1e-9

    def test_higher_fitness_gets_higher_weight(self):
        from gigaevo.adversarial.opponent_provider import _softmax_weights

        weights = _softmax_weights([0.1, 0.5, 0.9])
        assert weights[2] > weights[1] > weights[0]

    def test_negative_fitnesses_handled(self):
        from gigaevo.adversarial.opponent_provider import _softmax_weights

        weights = _softmax_weights([-1.0, 0.0, 1.0])
        assert abs(sum(weights) - 1.0) < 1e-9
        assert weights[2] > weights[0]

    @pytest.mark.asyncio
    async def test_nonfinite_fitness_fallback(self):
        """Non-finite fitness triggers uniform fallback."""
        import math
        import time

        provider = RedisOpponentArchiveProvider(
            host="localhost", port=6379, sources=[{"db": 1, "prefix": "test"}]
        )
        provider._cache = [
            OpponentProgram(program_id="p0", code="c0", fitness=math.nan),
            OpponentProgram(program_id="p1", code="c1", fitness=0.5),
            OpponentProgram(program_id="p2", code="c2", fitness=0.8),
            OpponentProgram(program_id="p3", code="c3", fitness=0.3),
            OpponentProgram(program_id="p4", code="c4", fitness=0.6),
            OpponentProgram(program_id="p5", code="c5", fitness=0.9),
        ]
        provider._cache_time = time.monotonic()
        result = await provider.get_opponents(3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: OpponentProgram dataclass
# ---------------------------------------------------------------------------


class TestOpponentProgram:
    def test_fields(self):
        p = OpponentProgram(program_id="abc123", code="x = 1", fitness=0.75)
        assert p.program_id == "abc123"
        assert p.code == "x = 1"
        assert p.fitness == 0.75
