"""Distributed instance lock with auto-renewal.

Prevents multiple instances from using the same Redis prefix. The
Redis-side semantics live in three Lua scripts under
``gigaevo/dataplane/scripts/`` (acquire / renew / release); each is
token-CAS, so renew and release only mutate the key when its stored
value still matches the holder's token.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import uuid

from loguru import logger

from gigaevo.database.redis.config import RedisLockConfig
from gigaevo.database.redis.connection import RedisConnection
from gigaevo.database.redis.keys import RedisProgramKeys
from gigaevo.dataplane.ids import ScriptName, make_script_name
from gigaevo.dataplane.scripts import load_lua_source
from gigaevo.exceptions import StorageError
from redis import asyncio as aioredis

_SCRIPT_LOCK_ACQUIRE: ScriptName = make_script_name("instance_lock_acquire")
_SCRIPT_LOCK_RENEW: ScriptName = make_script_name("instance_lock_renew")
_SCRIPT_LOCK_RELEASE: ScriptName = make_script_name("instance_lock_release")


class RedisInstanceLock:
    """Distributed instance lock with auto-renewal.

    Acquire stores the holder's instance id as the key's value; renew
    and release token-CAS against that value so a TTL-expired holder
    cannot operate on a successor's lock. Freshness is carried by the
    TTL alone, which means the stored value is stable across renewals.
    """

    def __init__(
        self,
        connection: RedisConnection,
        keys: RedisProgramKeys,
        config: RedisLockConfig,
    ):
        self._conn = connection
        self._keys = keys
        self._config = config

        self._instance_id = (
            f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        )
        self._token: str | None = None
        self._renewal_task: asyncio.Task[None] | None = None

        self._script_sources: dict[ScriptName, str] = {
            _SCRIPT_LOCK_ACQUIRE: load_lua_source(_SCRIPT_LOCK_ACQUIRE),
            _SCRIPT_LOCK_RENEW: load_lua_source(_SCRIPT_LOCK_RENEW),
            _SCRIPT_LOCK_RELEASE: load_lua_source(_SCRIPT_LOCK_RELEASE),
        }
        self._script_shas: dict[ScriptName, str] = {}
        self._reload_lock: asyncio.Lock | None = None

    @property
    def is_held(self) -> bool:
        return self._token is not None

    @property
    def instance_id(self) -> str:
        return self._instance_id

    # ── Lua plumbing ─────────────────────────────────────────────────

    def _get_reload_lock(self) -> asyncio.Lock:
        """Lazy-init the reload lock inside the running loop.

        Constructing the lock inside ``__init__`` would bind it to
        whichever loop happens to be current at object creation, which
        is not necessarily the loop that drives the renewer.
        """
        if self._reload_lock is None:
            self._reload_lock = asyncio.Lock()
        return self._reload_lock

    async def _load_sha(
        self,
        r: aioredis.Redis,
        name: ScriptName,
        *,
        stale_sha: str | None,
    ) -> str:
        """SCRIPT LOAD ``name`` and refresh the SHA cache, coalesced.

        Concurrent NOSCRIPT recoveries for the same script collapse to
        one SCRIPT LOAD round-trip: callers that wait on the lock find
        the SHA refreshed by the first arrival and reuse it. The
        ``stale_sha`` argument is the SHA that just raised NOSCRIPT;
        when it differs from the cache we are looking at a peer's
        already-installed result and skip the redundant LOAD.
        """
        async with self._get_reload_lock():
            cached = self._script_shas.get(name)
            if stale_sha is not None and cached is not None and cached != stale_sha:
                return cached
            loaded = await r.script_load(self._script_sources[name])  # type: ignore[misc]
            sha = loaded if isinstance(loaded, str) else loaded.decode("ascii")
            self._script_shas[name] = sha
            return sha

    async def _evalsha(
        self,
        r: aioredis.Redis,
        name: ScriptName,
        *,
        keys: list[str],
        args: list[str | int],
    ) -> int:
        """EVALSHA with one-shot NOSCRIPT reload-and-retry.

        Returns the script's integer return value (every instance-lock
        script returns 1 / 0). Raises whatever Redis errors propagate
        beyond the second attempt.
        """
        sha = self._script_shas.get(name)
        if sha is None:
            sha = await self._load_sha(r, name, stale_sha=None)
        try:
            return int(await r.evalsha(sha, len(keys), *keys, *args))  # type: ignore[misc]
        except aioredis.ResponseError as exc:
            if "NOSCRIPT" not in str(exc):
                raise
            sha = await self._load_sha(r, name, stale_sha=sha)
            return int(await r.evalsha(sha, len(keys), *keys, *args))  # type: ignore[misc]

    # ── public surface ───────────────────────────────────────────────

    async def acquire(self) -> bool:
        """Acquire the instance lock.

        Returns ``True`` on success; raises :class:`StorageError` if
        another instance holds the lock. The auto-renewal task is
        spawned on success.
        """

        async def _acquire(r: aioredis.Redis) -> bool:
            lock_key = self._keys.instance_lock()
            ttl_ms = max(1, int(self._config.lock_expiry_secs * 1000))
            ok = await self._evalsha(
                r,
                _SCRIPT_LOCK_ACQUIRE,
                keys=[lock_key],
                args=[self._instance_id, ttl_ms],
            )
            if ok != 1:
                existing = await r.get(lock_key)  # type: ignore[misc]
                raise StorageError(
                    f"Cannot start: another instance is using Redis prefix '{self._keys.prefix}'. "
                    f"Lock held by: {existing}. "
                    f"If this is a stale lock from a crashed instance, "
                    f"manually delete Redis key: {lock_key}"
                )
            logger.info(
                "[RedisInstanceLock] Acquired exclusive lock for prefix '{}'",
                self._keys.prefix,
            )
            self._token = self._instance_id
            return True

        result = await self._conn.execute("acquire_instance_lock", _acquire)
        self._renewal_task = asyncio.create_task(self._renew_periodically())
        return result

    async def release(self) -> None:
        """Release the instance lock.

        Token-CAS DEL: only deletes the key if its stored value still
        matches our instance id. Idempotent; an already-released or
        already-lost lock is a silent no-op.

        After ``release()`` returns the renewer is fully drained and the
        local holder state (``is_held``, ``_token``) is cleared, even if
        the Redis side errored — the caller's lock-holding posture ends
        at the call site regardless of network reachability so a follow-
        up acquire is not blocked by stale local state.
        """
        if self._renewal_task:
            self._renewal_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._renewal_task
            self._renewal_task = None

        if not self._token:
            return

        async def _release(r: aioredis.Redis) -> None:
            lock_key = self._keys.instance_lock()
            released = await self._evalsha(
                r,
                _SCRIPT_LOCK_RELEASE,
                keys=[lock_key],
                args=[self._instance_id],
            )
            if released == 1:
                logger.info(
                    "[RedisInstanceLock] Released lock for prefix '{}'",
                    self._keys.prefix,
                )

        try:
            await self._conn.execute("release_instance_lock", _release)
        except Exception as e:  # noqa: BLE001
            logger.warning("[RedisInstanceLock] Failed to release lock: {}", e)
        finally:
            self._token = None

    async def renew(self) -> bool:
        """Renew the instance lock.

        Token-CAS PEXPIRE: only refreshes the TTL if the stored value
        still matches our instance id. Returns ``False`` if the lock
        was lost (token mismatch or key absent).
        """
        if not self._token:
            return False

        async def _renew(r: aioredis.Redis) -> bool:
            lock_key = self._keys.instance_lock()
            ttl_ms = max(1, int(self._config.lock_expiry_secs * 1000))
            ok = await self._evalsha(
                r,
                _SCRIPT_LOCK_RENEW,
                keys=[lock_key],
                args=[self._instance_id, ttl_ms],
            )
            if ok != 1:
                logger.error(
                    "[RedisInstanceLock] Lost lock! Another instance may have taken over."
                )
                return False
            return True

        try:
            return await self._conn.execute("renew_instance_lock", _renew)
        except Exception as e:
            logger.error("[RedisInstanceLock] Failed to renew lock: {}", e)
            return False

    async def _renew_periodically(self) -> None:
        """Background task to renew the lock periodically."""
        while not self._conn.is_closing:
            try:
                await asyncio.sleep(self._config.lock_renewal_secs)

                if self._conn.is_closing:
                    break

                if not await self.renew():
                    logger.critical(
                        "[RedisInstanceLock] Failed to renew lock! "
                        "Another instance may be using the same prefix. STOPPING."
                    )
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[RedisInstanceLock] Renewal error: {}", e)
