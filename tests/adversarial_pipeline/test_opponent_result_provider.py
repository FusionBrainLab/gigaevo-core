"""Tests for OpponentResultProvider (exec + cached implementations).

The provider is the unit of "give me opponent evaluation payloads for these
IDs." Two first-class strategies, each correct for a different opponent
output shape:

  ExecOpponentResultProvider   — runs opponent code in a subprocess.
                                 Required when opponent entrypoint() returns
                                 a closure (G-side: D opponents are improvers
                                 whose improve(points) must be applied to the
                                 currently-evaluated G's point set).
  CachedOpponentResultProvider — reads stored CallProgramFunction output
                                 from opponents' Redis DBs (no exec).
                                 Correct when opponent output is a static
                                 value (D-side: G opponents are constructors
                                 whose entrypoint() returns a fixed ndarray).

Both must honour the same contract:
  produce(ids) -> list[Any | None]   (same length as ids; None on failure)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from gigaevo.adversarial.opponent_provider import (
    OpponentArchiveProvider,
    OpponentProgram,
)
from gigaevo.adversarial.opponent_result_provider import (
    CachedOpponentResultProvider,
    ExecOpponentResultProvider,
    OpponentResultProvider,
    build_opponent_result_provider,
)


class _FakeArchiveProvider(OpponentArchiveProvider):
    def __init__(self, opponents: list[OpponentProgram]):
        self._ops = opponents

    async def get_opponents(self, n: int = 5) -> list[OpponentProgram]:
        return self._ops[:n]

    async def get_top_k(
        self, k: int, *, higher_is_better: bool = True
    ) -> list[OpponentProgram]:
        return sorted(self._ops, key=lambda o: o.fitness, reverse=higher_is_better)[:k]

    async def get_programs_by_ids(self, ids: list[str]) -> list[OpponentProgram]:
        id_set = set(ids)
        return [o for o in self._ops if o.program_id in id_set]

    async def get_codes_by_ids(self, ids: list[str]) -> list[str]:
        m = {o.program_id: o.code for o in self._ops}
        return [m[i] for i in ids if i in m]


# ---------------------------------------------------------------------------
# ExecOpponentResultProvider
# ---------------------------------------------------------------------------


class TestExecOpponentResultProvider:
    @pytest.mark.asyncio
    async def test_produce_aligned_length(self):
        """len(produce(ids)) == len(ids), in order."""
        archive = _FakeArchiveProvider(
            [
                OpponentProgram(program_id="p1", code="c1", fitness=0.5),
                OpponentProgram(program_id="p2", code="c2", fitness=0.6),
            ]
        )
        provider = ExecOpponentResultProvider(
            archive_provider=archive,
            per_opponent_timeout=5.0,
            python_path=[],
            max_memory_mb=None,
        )

        async def mock_exec(**kwargs):
            return (f"out::{kwargs['code']}", b"", "")

        with patch(
            "gigaevo.adversarial.opponent_result_provider.run_exec_runner",
            side_effect=mock_exec,
        ):
            result = await provider.produce(["p1", "p2"])

        assert result == ["out::c1", "out::c2"]

    @pytest.mark.asyncio
    async def test_failed_opponent_becomes_none(self):
        """Exec failure => None placeholder at that index (not dropped)."""
        archive = _FakeArchiveProvider(
            [
                OpponentProgram(program_id="p1", code="good", fitness=0.5),
                OpponentProgram(program_id="p2", code="bad", fitness=0.6),
            ]
        )
        provider = ExecOpponentResultProvider(
            archive_provider=archive,
            per_opponent_timeout=5.0,
            python_path=[],
            max_memory_mb=None,
        )

        async def mock_exec(**kwargs):
            if kwargs["code"] == "bad":
                raise TimeoutError("boom")
            return ("good_result", b"", "")

        with patch(
            "gigaevo.adversarial.opponent_result_provider.run_exec_runner",
            side_effect=mock_exec,
        ):
            result = await provider.produce(["p1", "p2"])

        assert result == ["good_result", None]

    @pytest.mark.asyncio
    async def test_missing_id_becomes_none(self):
        """Id not in archive => None placeholder at that index."""
        archive = _FakeArchiveProvider(
            [OpponentProgram(program_id="p1", code="c1", fitness=0.5)]
        )
        provider = ExecOpponentResultProvider(
            archive_provider=archive,
            per_opponent_timeout=5.0,
            python_path=[],
            max_memory_mb=None,
        )

        async def mock_exec(**_kwargs):
            return ("x", b"", "")

        with patch(
            "gigaevo.adversarial.opponent_result_provider.run_exec_runner",
            side_effect=mock_exec,
        ):
            result = await provider.produce(["p1", "unknown"])

        assert result == ["x", None]

    @pytest.mark.asyncio
    async def test_produce_from_codes(self):
        """produce_from_codes bypasses archive lookup (used for fallback path)."""
        archive = _FakeArchiveProvider([])
        provider = ExecOpponentResultProvider(
            archive_provider=archive,
            per_opponent_timeout=5.0,
            python_path=[],
            max_memory_mb=None,
        )

        async def mock_exec(**kwargs):
            return (kwargs["code"].upper(), b"", "")

        with patch(
            "gigaevo.adversarial.opponent_result_provider.run_exec_runner",
            side_effect=mock_exec,
        ):
            result = await provider.produce_from_codes(["a", "b", "c"])

        assert result == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# CachedOpponentResultProvider
# ---------------------------------------------------------------------------


def _make_program_json(
    pid: str, *, stage_output: Any, status: str = "completed"
) -> str:
    """Build a Redis JSON payload mimicking RedisProgramStorage.add()."""
    from gigaevo.programs.utils import pickle_b64_serialize

    data = {
        "id": pid,
        "code": "irrelevant",
        "state": "evaluated",
        "stage_results": {
            "CallProgramFunction": {
                "status": status,
                "output": pickle_b64_serialize(stage_output)
                if stage_output is not None
                else None,
                "error": None,
            }
        },
    }
    return json.dumps(data)


class _StubRedis:
    """Minimal async-Redis stub supporting get() for program keys."""

    def __init__(self, blobs: dict[str, str]):
        self._blobs = blobs

    async def get(self, key: str):
        return self._blobs.get(key)

    async def mget(self, *keys: str):
        return [self._blobs.get(k) for k in keys]


class TestCachedOpponentResultProvider:
    @pytest.mark.asyncio
    async def test_fetch_stored_output(self):
        """Hit: returns the stored CallProgramFunction output."""
        blobs = {
            "pop_a:program:p1": _make_program_json("p1", stage_output=[[1, 2], [3, 4]]),
            "pop_a:program:p2": _make_program_json("p2", stage_output={"k": "v"}),
        }
        stub = _StubRedis(blobs)
        provider = CachedOpponentResultProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "pop_a"}],
        )
        with patch.object(provider, "_get_redis", return_value=stub):
            result = await provider.produce(["p1", "p2"])

        assert result == [[[1, 2], [3, 4]], {"k": "v"}]

    @pytest.mark.asyncio
    async def test_missing_id_returns_none(self):
        blobs = {"pop_a:program:p1": _make_program_json("p1", stage_output=42)}
        stub = _StubRedis(blobs)
        provider = CachedOpponentResultProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "pop_a"}],
        )
        with patch.object(provider, "_get_redis", return_value=stub):
            result = await provider.produce(["p1", "missing"])

        assert result == [42, None]

    @pytest.mark.asyncio
    async def test_failed_stage_returns_none(self):
        """Opponent was stored but its CallProgramFunction failed → None."""
        blobs = {
            "pop_a:program:p1": _make_program_json(
                "p1", stage_output=None, status="failed"
            )
        }
        stub = _StubRedis(blobs)
        provider = CachedOpponentResultProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "pop_a"}],
        )
        with patch.object(provider, "_get_redis", return_value=stub):
            result = await provider.produce(["p1"])

        assert result == [None]

    @pytest.mark.asyncio
    async def test_missing_stage_result_returns_none(self):
        """Program JSON has no CallProgramFunction entry → None."""
        data = {
            "id": "p1",
            "code": "x",
            "state": "evaluated",
            "stage_results": {},
        }
        blobs = {"pop_a:program:p1": json.dumps(data)}
        stub = _StubRedis(blobs)
        provider = CachedOpponentResultProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "pop_a"}],
        )
        with patch.object(provider, "_get_redis", return_value=stub):
            result = await provider.produce(["p1"])

        assert result == [None]

    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self):
        """Corrupt JSON in Redis → None (does not raise)."""
        blobs = {"pop_a:program:p1": "{not json"}
        stub = _StubRedis(blobs)
        provider = CachedOpponentResultProvider(
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "pop_a"}],
        )
        with patch.object(provider, "_get_redis", return_value=stub):
            result = await provider.produce(["p1"])

        assert result == [None]

    @pytest.mark.asyncio
    async def test_multi_source_first_hit_wins(self):
        """Id present in second source but not first is still found."""
        stub_src1 = _StubRedis({})  # empty
        stub_src2 = _StubRedis(
            {"pop_b:program:p1": _make_program_json("p1", stage_output="from_b")}
        )
        provider = CachedOpponentResultProvider(
            host="localhost",
            port=6379,
            sources=[
                {"db": 1, "prefix": "pop_a"},
                {"db": 2, "prefix": "pop_b"},
            ],
        )

        def fake_get_redis(db: int):
            return stub_src1 if db == 1 else stub_src2

        with patch.object(provider, "_get_redis", side_effect=fake_get_redis):
            result = await provider.produce(["p1"])

        assert result == ["from_b"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestBuildOpponentResultProvider:
    def test_exec_mode(self):
        archive = _FakeArchiveProvider([])
        provider = build_opponent_result_provider(
            mode="exec",
            archive_provider=archive,
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "p"}],
            per_opponent_timeout=10.0,
            python_path=[Path("/x")],
            max_memory_mb=None,
        )
        assert isinstance(provider, ExecOpponentResultProvider)

    def test_cached_mode(self):
        archive = _FakeArchiveProvider([])
        provider = build_opponent_result_provider(
            mode="cached",
            archive_provider=archive,
            host="localhost",
            port=6379,
            sources=[{"db": 1, "prefix": "p"}],
            per_opponent_timeout=10.0,
            python_path=[],
            max_memory_mb=None,
        )
        assert isinstance(provider, CachedOpponentResultProvider)

    def test_unknown_mode_raises(self):
        archive = _FakeArchiveProvider([])
        with pytest.raises(ValueError, match="opponent_result_mode"):
            build_opponent_result_provider(
                mode="teleport",  # type: ignore[arg-type]
                archive_provider=archive,
                host="localhost",
                port=6379,
                sources=[{"db": 1, "prefix": "p"}],
                per_opponent_timeout=10.0,
                python_path=[],
                max_memory_mb=None,
            )

    def test_both_impls_satisfy_interface(self):
        archive = _FakeArchiveProvider([])
        for mode in ("exec", "cached"):
            provider = build_opponent_result_provider(
                mode=mode,  # type: ignore[arg-type]
                archive_provider=archive,
                host="localhost",
                port=6379,
                sources=[{"db": 1, "prefix": "p"}],
                per_opponent_timeout=10.0,
                python_path=[],
                max_memory_mb=None,
            )
            assert isinstance(provider, OpponentResultProvider)


# Satisfy unused-import checks for mock AsyncMock when not used in assertions
_ = AsyncMock
