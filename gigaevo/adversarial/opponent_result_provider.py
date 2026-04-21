"""OpponentResultProvider — strategy for producing opponent evaluation payloads.

Two concrete strategies — both are first-class, each correct for a
different class of opponent output:

  ExecOpponentResultProvider
    Runs opponent `entrypoint()` in a fresh subprocess. This is the
    required path when the opponent output cannot be replayed from a
    cached value — e.g. when `entrypoint()` returns a closure whose
    behaviour depends on fresh per-call input (the Heilbronn G-side uses
    this: the opponents are D improvers whose `improve(points)` must run
    on the currently-evaluated G's point set).

  CachedOpponentResultProvider
    Reads the opponent's stored `CallProgramFunction` output from Redis.
    Correct when opponent output is a static value (ndarray, dict,
    scalar) already produced by the opponent's own run (the Heilbronn
    D-side uses this: opponents are G constructors whose `entrypoint()`
    returns a fixed point configuration). Program IDs are immutable by
    design, so the cached output is never stale.

Both strategies share a single contract:

    async def produce(ids: list[str]) -> list[Any | None]

The returned list is aligned index-wise with `ids`: each slot is either
the opponent's output or None (missing id, failed exec, corrupt blob,
stage not yet complete). Length preservation is a hard invariant —
DGTrackerStage and other downstream consumers rely on `len(results) ==
len(opponent_ids)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from redis import asyncio as aioredis

from gigaevo.adversarial.opponent_provider import OpponentArchiveProvider
from gigaevo.programs.stages.python_executors.wrapper import (
    ExecRunnerError,
    run_exec_runner,
)
from gigaevo.programs.utils import pickle_b64_deserialize


class OpponentResultProvider(ABC):
    """Produce evaluation payloads for a list of opponent program IDs."""

    @abstractmethod
    async def produce(self, ids: list[str]) -> list[Any | None]:
        """Return one result per id, in order. None marks failure/miss."""


class ExecOpponentResultProvider(OpponentResultProvider):
    """Run opponent code in a subprocess to recompute the entrypoint() output.

    First-class path for opponents whose output is not safely replayable
    from cache — e.g. closures. All sandboxing (memory cap, output cap,
    python_path) is preserved. On any of ExecRunnerError / TimeoutError /
    asyncio.CancelledError the slot becomes None so the caller always
    receives a fixed-length list aligned with the requested ids.
    """

    _OUTPUT_SIZE_CAP = 64 * 1024 * 1024

    def __init__(
        self,
        *,
        archive_provider: OpponentArchiveProvider,
        per_opponent_timeout: float,
        python_path: list[Path] | None,
        max_memory_mb: int | None,
    ):
        self._archive = archive_provider
        self._per_timeout = float(per_opponent_timeout)
        self._python_path = list(python_path or [])
        self._max_memory_mb = max_memory_mb

    async def produce(self, ids: list[str]) -> list[Any | None]:
        if not ids:
            return []
        # get_programs_by_ids returns OpponentProgram objects, so we can
        # build a pid → code map without worrying about positional alignment
        # (get_codes_by_ids would drop missing ids silently — bad for
        # length-preserving output).
        code_map = {
            p.program_id: p.code for p in await self._archive.get_programs_by_ids(ids)
        }
        tasks = [
            self._exec_one(code_map[pid]) if pid in code_map else self._none_task()
            for pid in ids
        ]
        raw = await asyncio.gather(*tasks, return_exceptions=False)
        ok = sum(1 for r in raw if r is not None)
        logger.info(
            "[ExecOpponentResultProvider] produced {}/{} opponents ok",
            ok,
            len(ids),
        )
        return list(raw)

    async def produce_from_codes(self, codes: list[str]) -> list[Any | None]:
        """Bypass archive lookup — execute the given raw codes in order.

        Used for the cold-start fallback path (archive empty, fall back to
        pre-loaded fallback_codes).
        """
        if not codes:
            return []
        raw = await asyncio.gather(*[self._exec_one(c) for c in codes])
        ok = sum(1 for r in raw if r is not None)
        logger.info(
            "[ExecOpponentResultProvider] fallback produced {}/{} ok",
            ok,
            len(codes),
        )
        return list(raw)

    async def _exec_one(self, code: str) -> Any | None:
        try:
            value, _, _ = await run_exec_runner(
                code=code,
                function_name="entrypoint",
                python_path=self._python_path,
                timeout=int(self._per_timeout),
                max_memory_mb=self._max_memory_mb,
                max_output_size=self._OUTPUT_SIZE_CAP,
            )
            return value
        except (ExecRunnerError, TimeoutError, asyncio.CancelledError) as exc:
            logger.debug("[ExecOpponentResultProvider] opponent exec failed: {}", exc)
            return None

    @staticmethod
    async def _none_task() -> None:
        return None


class CachedOpponentResultProvider(OpponentResultProvider):
    """Fetch opponent CallProgramFunction output from Redis without executing.

    Reads `{prefix}:program:{pid}` from each configured source; deserializes
    the pickled `stage_results["CallProgramFunction"].output`. Works across
    multiple sources (first hit wins per id).

    No sandboxing needed — the output was already produced inside the
    opponent's own sandboxed run.
    """

    _STAGE_NAME = "CallProgramFunction"
    _COMPLETED_STATUS = "completed"

    def __init__(
        self,
        *,
        host: str,
        port: int,
        sources: list[dict[str, int | str]],
    ):
        self._host = host
        self._port = port
        self._sources = [(int(s["db"]), str(s["prefix"])) for s in sources]
        self._clients: dict[int, aioredis.Redis] = {}

    def _get_redis(self, db: int) -> aioredis.Redis:
        if db not in self._clients:
            self._clients[db] = aioredis.Redis(
                host=self._host,
                port=self._port,
                db=db,
                decode_responses=True,
            )
        return self._clients[db]

    async def produce(self, ids: list[str]) -> list[Any | None]:
        if not ids:
            return []
        results: list[Any | None] = [None] * len(ids)
        for db, prefix in self._sources:
            r = self._get_redis(db)
            keys = [f"{prefix}:program:{pid}" for pid in ids]
            try:
                blobs = await r.mget(*keys)
            except Exception as exc:
                logger.warning(
                    "[CachedOpponentResultProvider] mget failed db={} err={}",
                    db,
                    exc,
                )
                continue
            for i, raw in enumerate(blobs):
                if results[i] is not None:
                    continue  # earlier source already supplied this slot
                if raw is None:
                    continue
                value = self._extract(raw, ids[i])
                if value is not None:
                    results[i] = value
        hits = sum(1 for r in results if r is not None)
        logger.info(
            "[CachedOpponentResultProvider] hit={} miss={} total={}",
            hits,
            len(ids) - hits,
            len(ids),
        )
        return results

    def _extract(self, raw: str, pid: str) -> Any | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "[CachedOpponentResultProvider] corrupt JSON pid={} err={}", pid, exc
            )
            return None
        stage_results = data.get("stage_results") or {}
        entry = stage_results.get(self._STAGE_NAME)
        if not isinstance(entry, dict):
            return None
        if entry.get("status") != self._COMPLETED_STATUS:
            return None
        out = entry.get("output")
        if out is None:
            return None
        try:
            box = pickle_b64_deserialize(out)
        except Exception as exc:
            logger.warning(
                "[CachedOpponentResultProvider] deserialize failed pid={} err={}",
                pid,
                exc,
            )
            return None
        # CallProgramFunction always wraps the raw value in a Box(data=...).
        # Unwrap so consumers see the same type as Exec mode returns.
        return getattr(box, "data", box)

    async def close(self) -> None:
        for client in self._clients.values():
            try:
                await client.close()
            except Exception:
                pass


def build_opponent_result_provider(
    mode: Literal["exec", "cached"],
    *,
    archive_provider: OpponentArchiveProvider,
    host: str,
    port: int,
    sources: list[dict[str, int | str]],
    per_opponent_timeout: float,
    python_path: list[Path] | None,
    max_memory_mb: int | None,
) -> OpponentResultProvider:
    """Factory — returns the provider matching `mode`.

    `mode="exec"`   → ExecOpponentResultProvider (always re-runs opponent code)
    `mode="cached"` → CachedOpponentResultProvider (reads opponent Redis)

    Unknown modes raise ValueError so mistyped config fails loudly at
    pipeline construction, not at the first DAG execution.
    """
    if mode == "exec":
        return ExecOpponentResultProvider(
            archive_provider=archive_provider,
            per_opponent_timeout=per_opponent_timeout,
            python_path=python_path,
            max_memory_mb=max_memory_mb,
        )
    if mode == "cached":
        return CachedOpponentResultProvider(
            host=host,
            port=port,
            sources=sources,
        )
    raise ValueError(
        f"unknown opponent_result_mode={mode!r}; expected 'exec' or 'cached'"
    )
