"""HTTP client factory for long-running asyncio paths.

Builds an ``aiohttp.ClientSession`` (and an ``openai.DefaultAioHttpClient``
adapter for the openai SDK) with defaults aimed at the failure surface
documented in upstream encode/httpx threads referenced from
gigaevo-core issue #9: long-running asyncio processes accumulate
connections that httpcore's async semaphore never releases, producing
silent ``PoolTimeout`` cascades on the timescale of hours. aiohttp's
connection pool is structurally different and survives that load
pattern; this factory pins the supporting defaults so the migration
buys what it should:

- Bounded ``ClientTimeout`` with LLM-friendly headroom — no ``None``
  pool waits, no silent forever-hang.
- Short HTTP ``keepalive_timeout`` — connections age out before any
  common load balancer's idle timeout (AWS ALB 60s, NGINX 75s, …)
  closes us asymmetrically and leaves a stale socket in the pool.
- TCP-level dead-peer detection via :data:`DEFAULT_SOCKET_OPTIONS` —
  applied to every transport after it's created (aiohttp's
  ``TCPConnector`` doesn't expose ``socket_options`` so the connector
  is subclassed in :class:`_KeepaliveConnector`).
- TLS 1.2+ minimum with hostname checking and CA verification, via
  :func:`gigaevo.infra._net.build_tls_context`.
- ``enable_cleanup_closed=True`` — aiohttp's own defense against the
  connection-leak class. No-op on Python 3.12+ where the SSL leak it
  countered was fixed upstream, but explicit so behavior is correct on
  older Python / aiohttp combos.
"""

from __future__ import annotations

import socket
import ssl
import sys
from typing import TYPE_CHECKING, Any

import aiohttp

from gigaevo.infra._net import apply_socket_options, build_tls_context, user_agent

if TYPE_CHECKING:
    from openai import DefaultAioHttpClient

# ``aiohttp.TCPConnector.enable_cleanup_closed=True`` previously countered
# CPython's SSL-socket close leak; the leak was fixed in CPython 3.12.13
# (PR python/cpython#118960), and aiohttp 3.13 emits a noisy warning when
# the kwarg is True on a runtime that already has the fix.  Only pass it
# on the affected runtime so we keep the leak defense without the noise.
_ENABLE_CLEANUP_CLOSED = sys.version_info < (3, 12, 13)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LIMIT = 300
DEFAULT_LIMIT_PER_HOST = 100
# Shorter than any common LB idle timeout (AWS ALB 60s, NGINX 75s,
# CloudFlare 90s). When the LB drops idle keepalive after its threshold,
# we'd otherwise hold a half-dead socket in the pool and burn the next
# request on it. Closing at 20s ages connections out before that race.
DEFAULT_KEEPALIVE_TIMEOUT = 20.0
DEFAULT_DNS_CACHE_TTL = 60

# LLM endpoints stream tokens for tens of seconds and occasionally
# minutes; sock_read must be generous. connect/sock_connect stay tight
# because a healthy handshake is fast and we want to fail-over rather
# than wait.
DEFAULT_TIMEOUT_TOTAL: float | None = 300.0
DEFAULT_TIMEOUT_CONNECT = 15.0
DEFAULT_TIMEOUT_SOCK_READ = 300.0
DEFAULT_TIMEOUT_SOCK_CONNECT = 15.0


# ---------------------------------------------------------------------------
# TCPConnector subclass: applies socket options after transport creation
# ---------------------------------------------------------------------------


class _KeepaliveConnector(aiohttp.TCPConnector):
    """``aiohttp.TCPConnector`` that applies :data:`DEFAULT_SOCKET_OPTIONS`.

    aiohttp doesn't expose ``socket_options`` on ``TCPConnector`` — the
    only way to set ``SO_KEEPALIVE`` etc. per connection is to intercept
    transport creation. We override ``_wrap_create_connection`` and,
    after the underlying ``create_connection`` returns, pull the socket
    via ``transport.get_extra_info('socket')`` and call setsockopt on
    each option. Failures are swallowed (see
    :func:`apply_socket_options`) so the call never fails an
    otherwise-good connection because the kernel rejected a tuning
    option.
    """

    async def _wrap_create_connection(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        transport, proto = await super()._wrap_create_connection(*args, **kwargs)
        sock = transport.get_extra_info("socket")
        if isinstance(sock, socket.socket):
            apply_socket_options(sock)
        return transport, proto


def build_connector(
    *,
    limit: int = DEFAULT_LIMIT,
    limit_per_host: int = DEFAULT_LIMIT_PER_HOST,
    keepalive_timeout: float = DEFAULT_KEEPALIVE_TIMEOUT,
    ttl_dns_cache: int = DEFAULT_DNS_CACHE_TTL,
    force_close: bool = False,
    ssl_context: ssl.SSLContext | None = None,
) -> aiohttp.TCPConnector:
    """Build a :class:`_KeepaliveConnector` with module defaults.

    ``enable_cleanup_closed=True`` is conditionally enabled — it was
    aiohttp's defense against CPython's SSL-socket close leak, and aiohttp
    3.13 emits a warning when the kwarg is True on a runtime that already
    has the upstream fix (CPython 3.12.13+).  See ``_ENABLE_CLEANUP_CLOSED``.

    ``ssl_context`` defaults to :func:`gigaevo.infra._net.build_tls_context`
    (TLS 1.2+, hostname + CA verification on).
    """
    kwargs: dict[str, Any] = dict(
        limit=limit,
        limit_per_host=limit_per_host,
        keepalive_timeout=keepalive_timeout,
        ttl_dns_cache=ttl_dns_cache,
        force_close=force_close,
        ssl=ssl_context if ssl_context is not None else build_tls_context(),
    )
    if _ENABLE_CLEANUP_CLOSED:
        kwargs["enable_cleanup_closed"] = True
    return _KeepaliveConnector(**kwargs)


def build_timeout(
    *,
    total: float | None = DEFAULT_TIMEOUT_TOTAL,
    connect: float | None = DEFAULT_TIMEOUT_CONNECT,
    sock_read: float | None = DEFAULT_TIMEOUT_SOCK_READ,
    sock_connect: float | None = DEFAULT_TIMEOUT_SOCK_CONNECT,
) -> aiohttp.ClientTimeout:
    """Build an ``aiohttp.ClientTimeout`` with bounded values.

    Every component is bounded by default. ``None`` is preserved when
    passed explicitly so callers can opt-out per-component (e.g. a
    streaming endpoint that legitimately needs unbounded ``sock_read``),
    but the defaults never produce a session that can hang silently.
    """
    return aiohttp.ClientTimeout(
        total=total,
        connect=connect,
        sock_read=sock_read,
        sock_connect=sock_connect,
    )


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def make_aiohttp_session(
    role: str,
    *,
    connector: aiohttp.TCPConnector | None = None,
    timeout: aiohttp.ClientTimeout | None = None,
    trust_env: bool = True,
    **session_kwargs: Any,
) -> aiohttp.ClientSession:
    """Create an ``aiohttp.ClientSession`` with module defaults.

    Args:
        role: Informational label (e.g. ``"memory_api"``,
            ``"llm_prompts"``). Reserved for per-role override hooks.
        connector: Optional override. When ``None``, :func:`build_connector`
            with the module defaults (TCP keepalive socket options and
            TLS 1.2+ context) is used.
        timeout: Optional override. When ``None``, :func:`build_timeout`
            with bounded defaults is used.
        trust_env: If ``True`` (default), aiohttp honors ``HTTP_PROXY`` /
            ``HTTPS_PROXY`` / ``NO_PROXY`` from the environment.
        **session_kwargs: Forwarded to ``aiohttp.ClientSession``.  A
            caller-supplied ``headers`` mapping is merged with our
            default ``User-Agent`` (caller wins on collision).

    Returns:
        Open ``aiohttp.ClientSession``. Caller owns the lifetime —
        ``await session.close()`` when done.
    """
    del role  # informational only
    headers: dict[str, str] = {"User-Agent": user_agent()}
    extra_headers = session_kwargs.pop("headers", None)
    if extra_headers:
        headers.update(extra_headers)
    return aiohttp.ClientSession(
        connector=connector if connector is not None else build_connector(),
        timeout=timeout if timeout is not None else build_timeout(),
        trust_env=trust_env,
        headers=headers,
        **session_kwargs,
    )


def make_openai_http_client(
    role: str,
    *,
    proxy: str | None = None,
    timeout: Any = None,
    **overrides: Any,
) -> DefaultAioHttpClient:
    """Create an aiohttp-backed ``http_client`` for the openai SDK.

    ``DefaultAioHttpClient`` is a thin wrapper around
    ``httpx_aiohttp.HttpxAiohttpClient`` (httpx surface, aiohttp
    transport).  It accepts httpx-shaped kwargs: ``timeout`` is
    ``httpx.Timeout``, ``headers`` is a mapping, ``proxy`` is a URL.

    The openai SDK's own default is ``httpx.Timeout(timeout=600,
    connect=5.0)`` — a 10-minute total per request, easy to hang a
    coroutine.  We override that to match the aiohttp factory:
    ``total=300s, connect=15s``.  Pass ``timeout=`` (an ``httpx.Timeout``)
    to override per-call.

    Args:
        role: Informational label.
        proxy: Optional proxy URL.
        timeout: Optional ``httpx.Timeout`` override.  Default is a
            bounded ``Timeout(timeout=300, connect=15)``.
        **overrides: Forwarded to ``DefaultAioHttpClient``.  A
            caller-supplied ``headers`` mapping is merged with our
            default ``User-Agent`` (caller wins on collision).

    Returns:
        ``openai.DefaultAioHttpClient`` instance.

    Raises:
        ImportError: When the ``openai[aiohttp]`` extra is missing. The
            error message includes the install instruction.
    """
    del role  # informational only
    try:
        from openai import DefaultAioHttpClient
    except ImportError as exc:
        raise ImportError(
            "openai.DefaultAioHttpClient is unavailable. Install the "
            "aiohttp extra: pip install 'openai[aiohttp]>=2.0.0'."
        ) from exc

    import httpx

    kwargs: dict[str, Any] = dict(overrides)
    if proxy is not None:
        kwargs.setdefault("proxy", proxy)
    if "timeout" not in kwargs:
        kwargs["timeout"] = (
            timeout
            if timeout is not None
            else httpx.Timeout(
                timeout=DEFAULT_TIMEOUT_TOTAL,
                connect=DEFAULT_TIMEOUT_CONNECT,
            )
        )
    headers: dict[str, str] = {"User-Agent": user_agent()}
    caller_headers = kwargs.get("headers")
    if caller_headers:
        headers.update(caller_headers)
    kwargs["headers"] = headers
    return DefaultAioHttpClient(**kwargs)
