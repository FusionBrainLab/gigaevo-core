# -*- coding: utf-8 -*-
"""Ultra-lightweight asynchronous Redis stub used for tests when a real Redis
or *fakeredis* is not available.  It implements only the methods required by
`RedisProgramStorage` and `RedisArchiveStorage` during unit/integration tests.

NOTE:  It is **NOT** a full Redis emulator; use only for single-process tests.
"""
from __future__ import annotations

import fnmatch
from typing import Any, Dict, List, Tuple

__all__ = ["FakeRedisStub"]


class FakePipeline:
    def __init__(self, parent: "FakeRedisStub") -> None:
        self._parent = parent
        self._cmds: List[Tuple[str, Tuple[Any, ...]]] = []

    # Redis string ops -----------------------------------------------------
    def get(self, key: str):  # noqa: D401 – interface parity
        self._cmds.append(("get", (key,)))

    def set(self, key: str, value: str):  # noqa: D401
        self._cmds.append(("set", (key, value)))

    # Redis stream / set ops added for pipeline compatibility -----------

    def xadd(self, key: str, fields, *args, **kwargs):  # noqa: D401
        # Ignore optional trimming args (maxlen, approximate)
        self._cmds.append(("xadd", (key, fields)))

    def sadd(self, key: str, member):  # noqa: D401
        self._cmds.append(("sadd", (key, member)))

    # Execution -----------------------------------------------------------
    async def execute(self):  # noqa: D401
        results = []
        for cmd, args in self._cmds:
            coro = getattr(self._parent, cmd)(*args)
            results.append(await coro)
        return results


class FakeRedisStub:  # pylint: disable=too-many-public-methods
    """Subset of aioredis API used by MetaEvolve tests."""

    def __init__(self) -> None:
        self._kv: Dict[str, str] = {}
        self._hash: Dict[str, Dict[str, str]] = {}
        self._set: Dict[str, set[str]] = {}

    # String commands -----------------------------------------------------
    async def get(self, key: str):
        return self._kv.get(key)

    async def set(self, key: str, value: str):
        self._kv[key] = value

    async def delete(self, key: str):
        self._kv.pop(key, None)
        self._hash.pop(key, None)
        self._set.pop(key, None)

    # Hash commands -------------------------------------------------------
    async def hset(self, key: str, mapping):
        self._hash.setdefault(key, {}).update(mapping)

    async def hgetall(self, key: str):
        return self._hash.get(key, {})

    # Set commands --------------------------------------------------------
    async def sadd(self, key: str, member):
        self._set.setdefault(key, set()).add(member)

    async def srem(self, key: str, member):
        if key in self._set:
            self._set[key].discard(member)

    async def smembers(self, key: str):
        return self._set.get(key, set())

    # Streams (simplified) ------------------------------------------------
    async def xadd(self, key: str, fields):
        return "0-1"  # fake ID

    # Scan ----------------------------------------------------------------
    async def scan_iter(self, pattern: str):
        for key in list(self._kv.keys()) + list(self._hash.keys()):
            if fnmatch.fnmatch(key, pattern):
                yield key

    # Transaction helpers -------------------------------------------------
    async def watch(self, *keys):
        # No concurrency – nothing to do
        return None

    async def unwatch(self):  # noqa: D401
        return None

    def multi_exec(self):  # noqa: D401 – mimic aioredis
        return self

    async def execute(self):  # part of pipeline API when acting as pipe
        return []

    # Pipeline ------------------------------------------------------------
    def pipeline(self):  # noqa: D401
        return FakePipeline(self)

    # Misc ----------------------------------------------------------------
    async def ping(self):
        return True

    async def flushdb(self):
        self._kv.clear()
        self._hash.clear()
        self._set.clear()

    async def close(self):
        pass

    # ------------------------------------------------------------------
    # Streams – blocking read (very minimal) ---------------------------
    # Added so the Runner can switch to event-driven mode even when using
    # the ultra-light FakeRedisStub.  We simply sleep for the requested
    # *block* timeout (milliseconds) and return an empty list.  This keeps
    # behaviour functionally equivalent to the previous polling sleep while
    # exercising the same code-path.
    # ------------------------------------------------------------------

    async def xread(self, streams, block: int = 0, count: int | None = None):  # noqa: D401
        if block:
            import asyncio

            await asyncio.sleep(block / 1000)
        return [] 