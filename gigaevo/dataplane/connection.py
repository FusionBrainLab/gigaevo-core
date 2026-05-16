"""Async Redis connection wrapper.

Owns exactly one ``redis.asyncio`` connection pool per :class:`DataPlane`
instance. Mandates ``decode_responses=True`` so every command returns
``str`` not ``bytes``.

Sync surfaces that need to talk to Redis go through ``asyncio.to_thread``
at the caller boundary; the dataplane does not expose a sync client.
"""

from __future__ import annotations

import asyncio
from typing import Final

from loguru import logger
import redis.asyncio as aioredis

from .errors import StartupError

_DEFAULT_MAX_CONNECTIONS: Final[int] = 64
_DEFAULT_SOCKET_TIMEOUT_S: Final[float] = 30.0
_DEFAULT_SOCKET_CONNECT_TIMEOUT_S: Final[float] = 10.0

# Upper bound on ``aclose`` during pool teardown. A server that has
# closed its TCP half but kept the socket alive can stall ``aclose``
# indefinitely; the bound below caps shutdown latency at a value that's
# still generous enough to drain a healthy pool.
_CLOSE_TIMEOUT_S: Final[float] = 5.0


class RedisConnection:
    """Lifecycle-managed async Redis pool.

    Construction is cheap (no I/O). Real connectivity happens in
    :meth:`startup` which PINGs the server and fails fast with a typed
    :class:`StartupError` on misconfiguration. :meth:`startup` is
    idempotent: a second call on an already-started instance is a
    no-op. Concurrent :meth:`startup` / :meth:`shutdown` invocations
    are serialised by an internal :class:`asyncio.Lock` so two racing
    callers cannot build two pools nor observe a torn lifecycle.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str,
        max_connections: int = _DEFAULT_MAX_CONNECTIONS,
        socket_timeout_s: float = _DEFAULT_SOCKET_TIMEOUT_S,
        socket_connect_timeout_s: float = _DEFAULT_SOCKET_CONNECT_TIMEOUT_S,
    ) -> None:
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._max_connections = max_connections
        self._socket_timeout_s = socket_timeout_s
        self._socket_connect_timeout_s = socket_connect_timeout_s
        self._pool: aioredis.Redis | None = None
        # Created lazily inside the running event loop so the connection
        # object remains safe to construct outside any loop context.
        self._lifecycle_lock: asyncio.Lock | None = None

    @property
    def key_prefix(self) -> str:
        return self._key_prefix

    @property
    def pool(self) -> aioredis.Redis:
        """The underlying async client. Only the coordinator should hold this."""
        if self._pool is None:
            raise RuntimeError(
                "RedisConnection.pool accessed before startup(); "
                "call await dp.startup() first"
            )
        return self._pool

    @property
    def is_started(self) -> bool:
        return self._pool is not None

    def _get_lifecycle_lock(self) -> asyncio.Lock:
        """Lazy-init the lock inside the running loop.

        Creating ``asyncio.Lock`` lazily avoids the historical "wrong
        loop" trap where a lock created at import time binds to the
        wrong loop and silently leaks.
        """
        if self._lifecycle_lock is None:
            self._lifecycle_lock = asyncio.Lock()
        return self._lifecycle_lock

    async def startup(self) -> None:
        """Build the pool and verify connectivity.

        Idempotent: a second call on a started instance returns
        immediately. Concurrent callers serialise on an internal lock;
        the first builds the pool, subsequent waiters short-circuit on
        the already-set ``_pool`` field. Raises :class:`StartupError`
        on failure and leaves ``self._pool`` unset (the partially-built
        pool is closed on the way out so the caller can safely retry
        without leaking sockets). Cancellation mid-PING also closes the
        partially-built pool — :class:`asyncio.CancelledError` is not
        an :class:`Exception` subclass and must be handled explicitly.
        """
        # Fast path with no I/O — re-checked under the lock below.
        if self._pool is not None:
            return
        async with self._get_lifecycle_lock():
            if self._pool is not None:
                return
            pool: aioredis.Redis | None = None
            try:
                pool = aioredis.Redis.from_url(
                    self._redis_url,
                    decode_responses=True,
                    max_connections=self._max_connections,
                    socket_timeout=self._socket_timeout_s,
                    socket_connect_timeout=self._socket_connect_timeout_s,
                )
                pong = await pool.ping()  # type: ignore[misc]
                if not pong:
                    raise StartupError(reason=f"PING returned falsey: {pong!r}")
            except StartupError:
                await _safe_close(pool)
                raise
            except asyncio.CancelledError:
                # CancelledError is BaseException — close the half-built
                # pool before propagating so the caller's retry sees a
                # clean slate.
                await _safe_close(pool)
                raise
            except Exception as exc:  # noqa: BLE001 - startup boundary, wrapped into StartupError
                await _safe_close(pool)
                raise StartupError(reason=f"Redis startup failed: {exc!r}") from exc
            self._pool = pool
            logger.info(
                "RedisConnection ready: url={} prefix={} pool={}",
                self._redis_url,
                self._key_prefix,
                self._max_connections,
            )

    async def shutdown(self) -> None:
        """Close the pool. Idempotent. Safe under partial-init failure.

        Serialised against :meth:`startup` via the same internal lock:
        a shutdown racing with an in-flight startup waits for that
        startup to either complete (and is then closed by this call)
        or fail (in which case ``_pool`` stays ``None`` and this is a
        no-op). The two cannot interleave to leave a dangling pool.
        """
        async with self._get_lifecycle_lock():
            pool = self._pool
            self._pool = None
            await _safe_close(pool)


async def _safe_close(pool: aioredis.Redis | None) -> None:
    """Best-effort close of a Redis client; never raises, bounded latency.

    ``aclose`` is wrapped in :func:`asyncio.wait_for` because a peer that
    half-closes the socket can otherwise stall the call forever and pin
    the entire shutdown path. A timeout-elapsed close still leaks the
    underlying socket, but the process can proceed; surface that as a
    warning so operators see the leak.
    """
    if pool is None:
        return
    try:
        await asyncio.wait_for(
            pool.aclose(),  # type: ignore[attr-defined]  # stub gap: aclose exists at runtime
            timeout=_CLOSE_TIMEOUT_S,
        )
    except TimeoutError:
        logger.warning(
            "RedisConnection close timed out after {}s; socket may leak",
            _CLOSE_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - shutdown boundary, never re-raised
        logger.warning("RedisConnection close swallowed error: {!r}", exc)


__all__ = ["RedisConnection"]
