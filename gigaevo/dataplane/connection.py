"""Async Redis connection wrapper.

Owns exactly one ``redis.asyncio`` connection pool per :class:`DataPlane`
instance. Mandates ``decode_responses=True`` so every command returns
``str`` not ``bytes`` — the bytes-vs-str confusion from the legacy
stream transport cannot recur.

No sync ``redis.Redis`` fallback inside the dataplane; sync surfaces
that still need to talk to Redis go through ``asyncio.to_thread`` at the
caller boundary until they are migrated.
"""

from __future__ import annotations

from typing import Final

from loguru import logger
import redis.asyncio as aioredis

from .errors import StartupError

_DEFAULT_MAX_CONNECTIONS: Final[int] = 64
_DEFAULT_SOCKET_TIMEOUT_S: Final[float] = 30.0
_DEFAULT_SOCKET_CONNECT_TIMEOUT_S: Final[float] = 10.0


class RedisConnection:
    """Lifecycle-managed async Redis pool.

    Construction is cheap (no I/O). Real connectivity happens in
    :meth:`startup` which PINGs the server and fails fast with a typed
    :class:`StartupError` on misconfiguration. :meth:`startup` is
    idempotent: a second call on an already-started instance is a
    no-op.
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

    async def startup(self) -> None:
        """Build the pool and verify connectivity.

        Idempotent: a second call on a started instance returns
        immediately. Raises :class:`StartupError` on failure and leaves
        ``self._pool`` unset (the partially-built pool is closed on the
        way out so the caller can safely retry without leaking sockets).
        """
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
        except Exception as exc:
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
        """Close the pool. Idempotent. Safe under partial-init failure."""
        pool = self._pool
        self._pool = None
        await _safe_close(pool)


async def _safe_close(pool: aioredis.Redis | None) -> None:
    """Best-effort close of a Redis client; never raises."""
    if pool is None:
        return
    try:
        await pool.aclose()  # type: ignore[attr-defined]  # stub gap: aclose exists at runtime
    except Exception as exc:  # noqa: BLE001
        logger.warning("RedisConnection close swallowed error: {!r}", exc)


__all__ = ["RedisConnection"]
