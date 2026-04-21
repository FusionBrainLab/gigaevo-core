"""Tests for gigaevo.adversarial.stages (two-stage DAG pattern).

Unit-level tests for FetchOpponentIdsStage and FetchOpponentResultsStage.
Provider-level behaviour (exec vs cached) lives in
test_opponent_result_provider.py; here we assert the stage coordinates the
provider correctly and handles cold-start fallback.
"""

from __future__ import annotations

from typing import Any

import pytest

from gigaevo.adversarial.opponent_provider import (
    OpponentArchiveProvider,
    OpponentProgram,
)
from gigaevo.adversarial.opponent_result_provider import (
    ExecOpponentResultProvider,
    OpponentResultProvider,
)
from gigaevo.adversarial.stages import FetchOpponentIdsStage, FetchOpponentResultsStage
from gigaevo.programs.program import Program
from gigaevo.programs.stages.common import Box

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeArchiveProvider(OpponentArchiveProvider):
    """Provider that returns pre-set opponents."""

    def __init__(self, opponents: list[OpponentProgram] | None = None):
        self._opponents = opponents or []

    async def get_opponents(self, n: int = 5) -> list[OpponentProgram]:
        return self._opponents[:n]

    async def get_top_k(
        self, k: int, *, higher_is_better: bool = True
    ) -> list[OpponentProgram]:
        return sorted(
            self._opponents, key=lambda o: o.fitness, reverse=higher_is_better
        )[:k]

    async def get_programs_by_ids(self, ids: list[str]) -> list[OpponentProgram]:
        id_set = set(ids)
        return [o for o in self._opponents if o.program_id in id_set]

    async def get_codes_by_ids(self, ids: list[str]) -> list[str]:
        id_map = {o.program_id: o.code for o in self._opponents}
        return [id_map[i] for i in ids if i in id_map]


class RecordingArchiveProvider(FakeArchiveProvider):
    """FakeArchiveProvider that records which sampler method was called.

    Used by sampling_mode routing tests to assert the stage picks the right
    provider entry point (get_top_k vs get_opponents) without asserting on
    stochastic output.
    """

    def __init__(self, opponents: list[OpponentProgram] | None = None):
        super().__init__(opponents)
        self.calls: list[tuple[str, int]] = []

    async def get_opponents(self, n: int = 5) -> list[OpponentProgram]:
        self.calls.append(("get_opponents", n))
        return await super().get_opponents(n)

    async def get_top_k(
        self, k: int, *, higher_is_better: bool = True
    ) -> list[OpponentProgram]:
        self.calls.append(("get_top_k", k))
        return await super().get_top_k(k, higher_is_better=higher_is_better)


class ScriptedResultProvider(OpponentResultProvider):
    """Provider driven by a {id: result_or_None} map. Returns aligned list."""

    def __init__(self, mapping: dict[str, Any] | None = None):
        self._map = mapping or {}
        self.calls: list[list[str]] = []

    async def produce(self, ids: list[str]) -> list[Any | None]:
        self.calls.append(list(ids))
        return [self._map.get(pid) for pid in ids]


def _make_program(code: str = "def entrypoint(): return 42") -> Program:
    return Program(code=code)


# ---------------------------------------------------------------------------
# FetchOpponentIdsStage
# ---------------------------------------------------------------------------


class TestFetchOpponentIdsStage:
    def test_uses_no_cache(self):
        from gigaevo.programs.stages.cache_handler import NO_CACHE

        stage = FetchOpponentIdsStage(
            opponent_provider=FakeArchiveProvider(), timeout=60.0
        )
        assert stage.cache_handler is NO_CACHE

    @pytest.mark.asyncio
    async def test_returns_box_of_ids(self):
        opponents = [
            OpponentProgram(program_id="p1", code="c1", fitness=0.5),
            OpponentProgram(program_id="p2", code="c2", fitness=0.8),
        ]
        stage = FetchOpponentIdsStage(
            opponent_provider=FakeArchiveProvider(opponents), timeout=60.0
        )
        result = await stage.compute(_make_program())
        assert isinstance(result, Box)
        assert set(result.data) == {"p1", "p2"}

    @pytest.mark.asyncio
    async def test_respects_n_opponents(self):
        opponents = [
            OpponentProgram(program_id=f"p{i}", code=f"c{i}", fitness=float(i))
            for i in range(10)
        ]
        stage = FetchOpponentIdsStage(
            opponent_provider=FakeArchiveProvider(opponents),
            n_opponents=3,
            timeout=60.0,
        )
        result = await stage.compute(_make_program())
        assert len(result.data) == 3

    # ------------------------------------------------------------------
    # sampling_mode ("top_k" default, "softmax" opt-in to reduce cache hits)
    # ------------------------------------------------------------------

    def test_invalid_sampling_mode_raises(self):
        with pytest.raises(ValueError, match="sampling_mode"):
            FetchOpponentIdsStage(
                opponent_provider=FakeArchiveProvider(),
                sampling_mode="bogus",
                timeout=60.0,
            )

    @pytest.mark.asyncio
    async def test_top_k_sampling_mode_is_default_and_deterministic(self):
        """Default stays top_k so repro-v1 and prior experiments are unchanged."""
        opponents = [
            OpponentProgram(program_id=f"p{i}", code=f"c{i}", fitness=float(i))
            for i in range(5)
        ]
        provider = FakeArchiveProvider(opponents)
        stage = FetchOpponentIdsStage(
            opponent_provider=provider, n_opponents=2, timeout=60.0
        )
        r1 = await stage.compute(_make_program())
        r2 = await stage.compute(_make_program())
        # Deterministic: two calls return the same top-2 ids in the same order.
        assert r1.data == r2.data
        # Top-2 by fitness → p4, p3.
        assert r1.data == ["p4", "p3"]

    @pytest.mark.asyncio
    async def test_softmax_sampling_mode_routes_to_get_opponents(self):
        """sampling_mode='softmax' must call provider.get_opponents(n), NOT get_top_k(n)."""
        opponents = [
            OpponentProgram(program_id=f"p{i}", code=f"c{i}", fitness=float(i))
            for i in range(5)
        ]
        provider = RecordingArchiveProvider(opponents)
        stage = FetchOpponentIdsStage(
            opponent_provider=provider,
            n_opponents=3,
            sampling_mode="softmax",
            timeout=60.0,
        )
        await stage.compute(_make_program())
        assert provider.calls == [("get_opponents", 3)]

    @pytest.mark.asyncio
    async def test_top_k_sampling_mode_routes_to_get_top_k(self):
        """sampling_mode='top_k' explicitly must call provider.get_top_k(n)."""
        opponents = [
            OpponentProgram(program_id=f"p{i}", code=f"c{i}", fitness=float(i))
            for i in range(5)
        ]
        provider = RecordingArchiveProvider(opponents)
        stage = FetchOpponentIdsStage(
            opponent_provider=provider,
            n_opponents=2,
            sampling_mode="top_k",
            timeout=60.0,
        )
        await stage.compute(_make_program())
        assert provider.calls == [("get_top_k", 2)]

    @pytest.mark.asyncio
    async def test_sampling_mode_accepts_enum_directly(self):
        """Enum value passes through without string detour."""
        from gigaevo.adversarial.opponent_provider import OpponentSamplingMode

        opponents = [
            OpponentProgram(program_id=f"p{i}", code=f"c{i}", fitness=float(i))
            for i in range(3)
        ]
        stage = FetchOpponentIdsStage(
            opponent_provider=FakeArchiveProvider(opponents),
            n_opponents=2,
            sampling_mode=OpponentSamplingMode.SOFTMAX,
            timeout=60.0,
        )
        result = await stage.compute(_make_program())
        assert len(result.data) == 2


# ---------------------------------------------------------------------------
# FetchOpponentResultsStage (thin coordinator over provider)
# ---------------------------------------------------------------------------


class TestFetchOpponentResultsStage:
    def test_default_cache_when_archive_reeval_true(self):
        from gigaevo.programs.stages.cache_handler import DEFAULT_CACHE

        stage = FetchOpponentResultsStage(
            result_provider=ScriptedResultProvider(),
            archive_reeval=True,
            timeout=60.0,
        )
        assert stage.get_cache_handler() is DEFAULT_CACHE

    def test_no_cache_when_archive_reeval_false(self):
        from gigaevo.programs.stages.cache_handler import NO_CACHE

        stage = FetchOpponentResultsStage(
            result_provider=ScriptedResultProvider(),
            archive_reeval=False,
            timeout=60.0,
        )
        assert stage.get_cache_handler() is NO_CACHE

    def test_fallback_without_exec_provider_raises(self):
        """Cached-only provider + fallback_codes must explicitly receive an
        exec provider for the cold-start path; fail loudly at construction."""

        # A provider that is NOT Exec — use the ScriptedResultProvider.
        with pytest.raises(ValueError, match="fallback_codes"):
            FetchOpponentResultsStage(
                result_provider=ScriptedResultProvider(),
                fallback_codes=["def entrypoint(): return 1"],
                timeout=60.0,
            )

    def test_fallback_reuses_exec_provider_when_available(self):
        archive = FakeArchiveProvider()
        exec_provider = ExecOpponentResultProvider(
            archive_provider=archive,
            per_opponent_timeout=5.0,
            python_path=[],
            max_memory_mb=None,
        )
        # Should not raise — Exec provider reused for fallback.
        FetchOpponentResultsStage(
            result_provider=exec_provider,
            fallback_codes=["def entrypoint(): return 1"],
            timeout=60.0,
        )

    @pytest.mark.asyncio
    async def test_empty_ids_no_fallback_returns_empty(self):
        provider = ScriptedResultProvider()
        stage = FetchOpponentResultsStage(result_provider=provider, timeout=60.0)
        stage.attach_inputs({"opponent_ids": Box[object](data=[])})
        result = await stage.compute(_make_program())
        assert result.data == []
        # Provider was still consulted with empty list.
        assert provider.calls == [[]]

    @pytest.mark.asyncio
    async def test_delegates_to_provider(self):
        provider = ScriptedResultProvider({"p1": "r1", "p2": "r2"})
        stage = FetchOpponentResultsStage(result_provider=provider, timeout=60.0)
        stage.attach_inputs({"opponent_ids": Box[object](data=["p1", "p2"])})
        result = await stage.compute(_make_program())
        assert result.data == ["r1", "r2"]
        assert provider.calls == [["p1", "p2"]]

    @pytest.mark.asyncio
    async def test_none_slots_preserved(self):
        """Provider's None placeholders must flow through unchanged."""
        provider = ScriptedResultProvider({"p1": "r1"})  # p2 missing -> None
        stage = FetchOpponentResultsStage(result_provider=provider, timeout=60.0)
        stage.attach_inputs({"opponent_ids": Box[object](data=["p1", "p2"])})
        result = await stage.compute(_make_program())
        assert result.data == ["r1", None]

    @pytest.mark.asyncio
    async def test_cold_start_fallback_triggers_exec(self):
        """Empty ids + fallback_codes → fallback path runs via exec provider."""
        archive = FakeArchiveProvider()
        exec_provider = ExecOpponentResultProvider(
            archive_provider=archive,
            per_opponent_timeout=5.0,
            python_path=[],
            max_memory_mb=None,
        )
        recorded: list[str] = []

        async def fake_produce_from_codes(codes):
            recorded.extend(codes)
            return [f"ran::{c}" for c in codes]

        exec_provider.produce_from_codes = fake_produce_from_codes  # type: ignore[assignment]

        stage = FetchOpponentResultsStage(
            result_provider=exec_provider,
            fallback_codes=["code_a", "code_b"],
            timeout=60.0,
        )
        stage.attach_inputs({"opponent_ids": Box[object](data=[])})
        result = await stage.compute(_make_program())
        assert result.data == ["ran::code_a", "ran::code_b"]
        assert recorded == ["code_a", "code_b"]

    @pytest.mark.asyncio
    async def test_all_none_triggers_fallback(self):
        """Provider returns all Nones → fallback still kicks in."""

        class AllNone(OpponentResultProvider):
            async def produce(self, ids: list[str]) -> list[Any | None]:
                return [None] * len(ids)

        archive = FakeArchiveProvider()
        exec_provider = ExecOpponentResultProvider(
            archive_provider=archive,
            per_opponent_timeout=5.0,
            python_path=[],
            max_memory_mb=None,
        )

        async def fake_produce_from_codes(codes):
            return [f"fb::{c}" for c in codes]

        exec_provider.produce_from_codes = fake_produce_from_codes  # type: ignore[assignment]

        stage = FetchOpponentResultsStage(
            result_provider=AllNone(),
            fallback_codes=["code_x"],
            fallback_exec_provider=exec_provider,
            timeout=60.0,
        )
        stage.attach_inputs({"opponent_ids": Box[object](data=["p1", "p2"])})
        result = await stage.compute(_make_program())
        assert result.data == ["fb::code_x"]
