"""Lua script registry.

Lifted from the proven pattern in :mod:`gigaevo.infra.endpoint_pool`:
load every script at startup, cache its SHA, EVALSHA on the hot path,
on ``NOSCRIPT`` reload-and-retry exactly once. A second consecutive
``NOSCRIPT`` raises :class:`ScriptLostError` — that should not happen,
and we want loud failure if it does.

Scripts are registered by name via :meth:`register`, loaded into Redis
en masse via :meth:`load_all` at coordinator startup, and invoked by
name via :meth:`evalsha`. This module ships without any scripts
registered; each per-resource module registers its own during
coordinator startup.

Concurrency contract: every coroutine method takes an internal lock on
the SHA cache so the NOSCRIPT recovery path stays atomic — two callers
hitting NOSCRIPT for the same script at the same time will reload
exactly once, not twice. Reads of ``registered`` / ``loaded_count`` /
``is_registered`` do not take the lock; they are best-effort snapshots
used in logging and tests.
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

        The script is not yet loaded into Redis; :meth:`load_all` does
        that. Re-registering an existing name overwrites — useful in
        tests that swap scripts, mildly surprising in production.
        Overwriting also invalidates any cached SHA for ``name`` so the
        next :meth:`evalsha` reloads against the new source instead of
        EVALSHA-ing the stale SHA and getting a confusing miss.

        Registration after :meth:`load_all` is supported: the missing
        SHA path inside :meth:`evalsha` will SCRIPT LOAD on first use.
        """
        existing = self._scripts.get(name)
        if existing is not None and existing != source:
            logger.warning(
                "LuaRegistry.register: overwriting existing script {!r}", name
            )
            # The previously cached SHA refers to the old source. Drop
            # it so the next evalsha reloads against ``source`` rather
            # than EVALSHA-ing a SHA that does not match the script the
            # caller just installed.
            self._sha.pop(name, None)
        self._scripts[name] = source

    def is_registered(self, name: ScriptName) -> bool:
        """True if ``name`` has a source — irrespective of SHA-cache state."""
        return name in self._scripts

    async def load_all(self) -> None:
        """SCRIPT LOAD every registered script; cache the SHAs.

        On failure of any individual load, the SHAs cached so far are
        retained but the caller's startup will surface the exception —
        partial load leaves the registry in a "some loaded, some not"
        state that subsequent :meth:`evalsha` calls will repair via the
        NOSCRIPT reload path. Calls to :meth:`register` made between
        partial load failure and retry are picked up by the next
        :meth:`load_all`.
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
            # The ``stale_sha`` we just EVALSHA-d is what NOSCRIPT
            # rejected; pass it so concurrent NOSCRIPT recoveries that
            # already refreshed the cache short-circuit instead of
            # issuing a redundant SCRIPT LOAD.
            sha = await self._reload(name, stale_sha=sha)
            try:
                return await self._redis.evalsha(sha, len(keys), *keys, *args)  # type: ignore[misc]
            except NoScriptError as exc:
                raise ScriptLostError(script_name=name) from exc

    async def _reload(self, name: ScriptName, *, stale_sha: str | None) -> str:
        """Re-issue SCRIPT LOAD for ``name`` and refresh the SHA cache.

        Serialised so two callers racing for the same name produce
        exactly one SCRIPT LOAD round-trip. Two entry shapes share this
        method:

            - First-time load (``stale_sha is None``): the call entered
              ``_reload`` because the SHA cache had no entry for
              ``name``. After acquiring the lock the holder re-reads
              the cache; if a peer populated it during the wait we
              reuse that entry instead of issuing a redundant
              ``script_load``.
            - NOSCRIPT recovery (``stale_sha`` is a value): the lock
              holder compares the cache against ``stale_sha``. A
              differing cache means a peer already refreshed; reuse
              that entry. A matching cache means the dead SHA is still
              the latest, so issue a fresh ``script_load``.

        Both shapes collapse concurrent reloaders to a single Redis
        round-trip and preserve the single-caller invariant that
        ``evalsha`` advances past NOSCRIPT instead of EVALSHA-ing the
        same dead SHA forever.
        """
        async with self._get_reload_lock():
            cached = self._sha.get(name)
            if cached is not None and cached != stale_sha:
                # Either a first-load peer populated the cache while we
                # waited (stale_sha is None, cached is set) or a NOSCRIPT
                # peer already refreshed (stale_sha != cached). Either
                # way the cached SHA is fresh.
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
