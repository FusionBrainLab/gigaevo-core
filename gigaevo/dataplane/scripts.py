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
"""

from __future__ import annotations

from typing import Any

from loguru import logger
import redis.asyncio as aioredis
from redis.exceptions import NoScriptError

from .errors import ScriptLostError, ScriptNotRegisteredError
from .ids import ScriptName


class LuaRegistry:
    """Holds Lua script sources, SHAs, and an evalsha wrapper."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._scripts: dict[ScriptName, str] = {}
        self._sha: dict[ScriptName, str] = {}

    def register(self, name: ScriptName, source: str) -> None:
        """Store a Lua source under a logical name.

        The script is not yet loaded into Redis; :meth:`load_all` does
        that. Re-registering an existing name overwrites — useful in
        tests that swap scripts, mildly surprising in production. If
        :meth:`load_all` has already run, the new source will not be
        live until a subsequent :meth:`load_all` (or the next NOSCRIPT
        reload triggered by :meth:`evalsha`).
        """
        if name in self._scripts and self._scripts[name] != source:
            logger.warning(
                "LuaRegistry.register: overwriting existing script {!r}", name
            )
        self._scripts[name] = source

    async def load_all(self) -> None:
        """SCRIPT LOAD every registered script; cache the SHAs.

        On failure of any individual load, any SHAs cached so far are
        retained but the caller's startup will surface the exception —
        partial load leaves the registry in a "some loaded, some not"
        state that subsequent :meth:`evalsha` calls will repair via the
        NOSCRIPT reload path.
        """
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
        ``NOSCRIPT`` is observed.
        """
        if name not in self._scripts:
            raise ScriptNotRegisteredError(script_name=name)
        sha = self._sha.get(name)
        if sha is None:
            sha = await self._reload(name)
        try:
            return await self._redis.evalsha(sha, len(keys), *keys, *args)  # type: ignore[misc]
        except NoScriptError:
            sha = await self._reload(name)
            try:
                return await self._redis.evalsha(sha, len(keys), *keys, *args)  # type: ignore[misc]
            except NoScriptError as exc:
                raise ScriptLostError(script_name=name) from exc

    async def _reload(self, name: ScriptName) -> str:
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


__all__ = ["LuaRegistry"]
