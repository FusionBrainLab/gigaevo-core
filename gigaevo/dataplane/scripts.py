"""Lua script registry.

Load every script at startup, cache its SHA, EVALSHA on the hot path,
on ``NOSCRIPT`` reload-and-retry exactly once. A second consecutive
``NOSCRIPT`` raises :class:`ScriptLostError`.

Concurrency: coroutine methods take an internal lock on the SHA cache
so the NOSCRIPT recovery path is atomic — concurrent callers hitting
NOSCRIPT for the same script reload exactly once. Reads of
``registered`` / ``loaded_count`` / ``is_registered`` are lockless
snapshots for logging and tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger
import redis.asyncio as aioredis
from redis.exceptions import NoScriptError

from .errors import ScriptLostError, ScriptNotRegisteredError
from .ids import ScriptName

_SCRIPTS_DIR = Path(__file__).parent / "scripts"


def load_lua_source(name: ScriptName) -> str:
    """Read a ``.lua`` source from the ``gigaevo/dataplane/scripts/`` directory.

    Looks up ``<name>.lua`` relative to the package's ``scripts/`` folder.
    Raises :class:`FileNotFoundError` if the file does not exist — that's
    a packaging or naming bug, not a runtime condition; we surface it as
    a plain ``FileNotFoundError`` so it fails loudly at startup.
    """
    path = _SCRIPTS_DIR / f"{name}.lua"
    return path.read_text(encoding="utf-8")


class LuaRegistry:
    """Holds Lua script sources, SHAs, and an evalsha wrapper."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._scripts: dict[ScriptName, str] = {}
        self._sha: dict[ScriptName, str] = {}
        self._reload_lock: asyncio.Lock | None = None

    def _get_reload_lock(self) -> asyncio.Lock:
        """Lazy-init the lock inside the running loop."""
        if self._reload_lock is None:
            self._reload_lock = asyncio.Lock()
        return self._reload_lock

    def register(self, name: ScriptName, source: str) -> None:
        """Store a Lua source under a logical name.

        Not loaded into Redis until :meth:`load_all`. Re-registering an
        existing name overwrites and invalidates any cached SHA so the
        next :meth:`evalsha` reloads against the new source. Registration
        after :meth:`load_all` is supported via the SHA-miss path.
        """
        existing = self._scripts.get(name)
        if existing is not None and existing != source:
            logger.warning(
                "LuaRegistry.register: overwriting existing script {!r}", name
            )
            # Drop the SHA so evalsha reloads against the new source.
            self._sha.pop(name, None)
        self._scripts[name] = source

    def is_registered(self, name: ScriptName) -> bool:
        """True if ``name`` has a source — irrespective of SHA-cache state."""
        return name in self._scripts

    async def load_all(self) -> None:
        """SCRIPT LOAD every registered script; cache the SHAs.

        Partial-load failures surface to the caller; already-cached SHAs
        are retained and any missing ones repair through the NOSCRIPT
        path on first :meth:`evalsha`.
        """
        async with self._get_reload_lock():
            for name, source in self._scripts.items():
                sha = await self._redis.script_load(source)  # type: ignore[misc]
                self._sha[name] = sha if isinstance(sha, str) else sha.decode("ascii")
        if self._scripts:
            logger.info(
                "LuaRegistry loaded {} scripts: {}",
                len(self._scripts),
                sorted(self._scripts),
            )

    async def evalsha(
        self,
        name: ScriptName,
        *,
        keys: list[str],
        args: list[str | int],
    ) -> Any:
        """Invoke a registered script. On ``NOSCRIPT``, reload once and retry.

        Raises :class:`ScriptLostError` if a second consecutive
        ``NOSCRIPT`` is observed (Redis is in a bad state and the caller
        should not paper over it). Raises
        :class:`ScriptNotRegisteredError` if ``name`` has no registered
        source — that is a programming error, not a transient fault.
        """
        if name not in self._scripts:
            raise ScriptNotRegisteredError(script_name=name)
        sha = self._sha.get(name)
        if sha is None:
            sha = await self._reload(name, stale_sha=None)
        try:
            return await self._redis.evalsha(sha, len(keys), *keys, *args)  # type: ignore[misc]
        except NoScriptError:
            # Pass the rejected SHA so concurrent reloaders short-circuit
            # if a peer already refreshed the cache.
            sha = await self._reload(name, stale_sha=sha)
            try:
                return await self._redis.evalsha(sha, len(keys), *keys, *args)  # type: ignore[misc]
            except NoScriptError as exc:
                raise ScriptLostError(script_name=name) from exc

    async def _reload(self, name: ScriptName, *, stale_sha: str | None) -> str:
        """Re-issue SCRIPT LOAD for ``name`` and refresh the SHA cache.

        Serialised so concurrent callers produce one SCRIPT LOAD. Two
        entry shapes: first-time load (``stale_sha is None``) and
        NOSCRIPT recovery. Both reuse a peer-refreshed cache entry when
        present (``cached != stale_sha``) instead of issuing a duplicate
        load.
        """
        async with self._get_reload_lock():
            cached = self._sha.get(name)
            if cached is not None and cached != stale_sha:
                return cached
            source = self._scripts[name]
            sha = await self._redis.script_load(source)  # type: ignore[misc]
            sha_str = sha if isinstance(sha, str) else sha.decode("ascii")
            self._sha[name] = sha_str
            return sha_str

    @property
    def registered(self) -> tuple[ScriptName, ...]:
        return tuple(self._scripts)

    @property
    def loaded_count(self) -> int:
        """Number of scripts whose SHA is cached locally."""
        return len(self._sha)


__all__ = ["LuaRegistry", "load_lua_source"]
