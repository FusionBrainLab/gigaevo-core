"""Distributed instance lock with auto-renewal.

Prevents multiple instances from using the same Redis prefix. The
Redis-side semantics live in three Lua scripts under
``gigaevo/dataplane/scripts/`` (acquire / renew / release); each is
token-CAS, so renew and release only mutate the key when its stored
value still matches the holder's token.

Two execution paths share the same lock-key shape:

* **Dataplane-routed**: when a :class:`~gigaevo.dataplane.DataPlane` is
  supplied at construction, the three Lua calls dispatch through
  :meth:`gigaevo.dataplane.scripts.LuaRegistry.evalsha` so the SHA
  cache, NOSCRIPT recovery, and :class:`ScriptLostError` escalation
  policy are owned by a single registry shared with every other
  coordinator-routed caller. The acquired lock is wrapped in a
  :class:`~gigaevo.dataplane.crash.CrashWatchedHandle` so renewal
  failure surfaces as a typed :class:`~gigaevo.dataplane.crash.CrashEvent`
  the caller can pattern-match against, instead of forcing readers to
  poll a private flag.

* **Legacy direct-aioredis**: when ``dataplane`` is ``None`` the lock
  reimplements ``SCRIPT LOAD`` + ``EVALSHA`` + ``NOSCRIPT`` retry
  locally. This path stays in place for callers that have not yet
  acquired a dataplane handle.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
from time import monotonic
import uuid

from loguru import logger

from gigaevo.database.redis.config import RedisLockConfig
from gigaevo.database.redis.connection import RedisConnection
from gigaevo.database.redis.keys import RedisProgramKeys
from gigaevo.dataplane.coordinator import DataPlane, InstanceLease
from gigaevo.dataplane.crash import CrashEvent, CrashWatchedHandle, OneShotFlag
from gigaevo.dataplane.ids import LeaseToken, ScriptName, make_script_name
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
        *,
        dataplane: DataPlane | None = None,
    ):
        self._conn = connection
        self._keys = keys
        self._config = config
        self._dataplane = dataplane

        self._instance_id = (
            f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        )
        self._token: str | None = None
        self._renewal_task: asyncio.Task[None] | None = None

        # Legacy direct-aioredis SHA cache. Populated lazily on the
        # first :meth:`_evalsha` call when ``self._dataplane is None``;
        # ignored on the dataplane-routed path where
        # :class:`LuaRegistry` owns the cache.
        self._script_sources: dict[ScriptName, str] = {
            _SCRIPT_LOCK_ACQUIRE: load_lua_source(_SCRIPT_LOCK_ACQUIRE),
            _SCRIPT_LOCK_RENEW: load_lua_source(_SCRIPT_LOCK_RENEW),
            _SCRIPT_LOCK_RELEASE: load_lua_source(_SCRIPT_LOCK_RELEASE),
        }
        self._script_shas: dict[ScriptName, str] = {}
        self._reload_lock: asyncio.Lock | None = None

        # Typed-control-flow channel for renewal-loss observability.
        # Populated on :meth:`acquire`; the flag is signalled by the
        # background renewer on token-CAS failure or transport error.
        # Callers route lease-scoped operations through
        # ``self._wrapped_lease.call(...)`` to convert the polled flag
        # into a :class:`CrashEvent` branch.
        self._lease_flag: OneShotFlag | None = None
        self._wrapped_lease: (
            CrashWatchedHandle[InstanceLease, str, InstanceLease] | None
        ) = None

    @property
    def is_held(self) -> bool:
        return self._token is not None

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def wrapped_lease(
        self,
    ) -> CrashWatchedHandle[InstanceLease, str, InstanceLease] | None:
        """Typed crash-watched handle around the held lease.

        ``None`` when the lock is not held or when the dataplane is not
        wired (the legacy path has no shared :class:`InstanceLease`
        vocabulary). On the dataplane-routed path the handle's
        :meth:`CrashWatchedHandle.call` short-circuits to a
        :class:`CrashEvent` once the background renewer has signalled
        renewal loss, eliminating a Redis round-trip for the loss-detection
        path.
        """
        return self._wrapped_lease

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

        Used only on the legacy direct-aioredis path.
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
        """Invoke a lock script and coerce its return to ``int``.

        Two dispatch paths, selected by whether the dataplane is wired:

        * **Dataplane-routed**: delegates to
          :meth:`LuaRegistry.evalsha`, which owns the SHA cache,
          NOSCRIPT-reload retry, and :class:`ScriptLostError` escalation
          policy. ``r`` is ignored.
        * **Legacy**: replicates the SCRIPT LOAD + EVALSHA + NOSCRIPT
          retry locally against the per-lock SHA cache.

        Returns the script's integer return value (every instance-lock
        script returns 1 / 0).
        """
        if self._dataplane is not None:
            lua = self._dataplane._lua
            if lua is None:
                raise StorageError(
                    "RedisInstanceLock: dataplane was supplied but its "
                    "LuaRegistry is not initialised; call DataPlane.startup() "
                    "before constructing the lock."
                )
            return int(await lua.evalsha(name, keys=keys, args=args))

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
        spawned on success. When the dataplane is wired, the acquired
        lock is wrapped in a :class:`CrashWatchedHandle` for typed
        renewal-loss observability via :attr:`wrapped_lease`.
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
        if self._dataplane is not None:
            # Build an InstanceLease-shaped record so the standard
            # dp.wrap_lease policy applies. The lease's ``token`` is the
            # holder's instance id (the same value token-CAS'd by the
            # underlying scripts); ``flag`` is signalled by the
            # background renewer below on renewal loss.
            self._lease_flag = OneShotFlag()
            lease = InstanceLease(
                token=LeaseToken(self._instance_id),
                key=self._keys.instance_lock(),
                ttl_s=float(self._config.lock_expiry_secs),
                expires_at_monotonic=(
                    monotonic() + float(self._config.lock_expiry_secs)
                ),
                flag=self._lease_flag,
            )
            self._wrapped_lease = self._dataplane.wrap_lease(lease)
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
            self._wrapped_lease = None
            self._lease_flag = None

    async def renew(self) -> bool:
        """Renew the instance lock.

        Token-CAS PEXPIRE: only refreshes the TTL if the stored value
        still matches our instance id. Returns ``False`` if the lock
        was lost (token mismatch or key absent); signals the crash flag
        on loss so observers polling :attr:`wrapped_lease` short-circuit
        to a :class:`CrashEvent` on the next call.
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
            ok = await self._conn.execute("renew_instance_lock", _renew)
        except Exception as e:
            logger.error("[RedisInstanceLock] Failed to renew lock: {}", e)
            if self._lease_flag is not None:
                self._lease_flag.signal()
            return False
        if not ok and self._lease_flag is not None:
            self._lease_flag.signal()
        return ok

    async def observe_loss(
        self,
    ) -> CrashEvent[str, InstanceLease] | None:
        """Return a :class:`CrashEvent` iff the renewer has signalled loss.

        ``None`` on the non-loss path (the typical case). The first call
        after a loss returns a synthesized :class:`CrashEvent` carrying
        the lock key as ``peer`` and the lost :class:`InstanceLease` as
        ``resource``. Subsequent calls return ``None`` so observers do
        not see the same loss twice.

        Returns ``None`` unconditionally on the legacy direct-aioredis
        path where ``wrapped_lease`` is unavailable; callers that need
        crash-event observability MUST construct the lock with a
        dataplane handle.
        """
        if self._wrapped_lease is None:
            return None

        async def _noop(_: InstanceLease) -> bool:
            return True

        _, evt = await self._wrapped_lease.call(_noop)
        if evt is not None:
            # Consume the loss: clear the wrapper so a second observer
            # does not see the same event a second time.
            self._wrapped_lease = None
            self._lease_flag = None
        return evt

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
