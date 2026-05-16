"""``requests.Session`` factory for sync HTTP paths.

The sync counterpart to :mod:`gigaevo.infra.aiohttp_factory`. Builds a
``requests.Session`` whose underlying ``urllib3`` connection pool is
explicitly sized, whose retry policy is wired through
``urllib3.util.retry.Retry`` (retries happen inside ``urllib3`` before
the response reaches application code), whose sockets carry
``SO_KEEPALIVE`` with explicit per-OS probe timing for fast dead-peer
detection, and whose TLS context pins minimum TLS 1.2 with hostname
verification.

A thin ``Session`` subclass injects a default per-request timeout split
into a ``(connect, read)`` tuple. ``requests`` has no session-level
timeout knob; without this any unset ``timeout=`` lets a hung
connection block indefinitely.

A small ``HTTPAdapter`` subclass forwards ``socket_options`` and the
TLS context to ``urllib3.PoolManager`` so the keepalive + TLS settings
apply to every connection minted by the pool.
"""

from __future__ import annotations

import ssl
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry

from gigaevo.infra._net import DEFAULT_SOCKET_OPTIONS, build_tls_context, user_agent

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_POOL_CONNECTIONS = 20
DEFAULT_POOL_MAXSIZE = 100

# Split timeout: connect must be short (a healthy TCP handshake is fast,
# but DNS / TLS handshake variability needs headroom); read is generous so
# LLM-style endpoints with slow first-byte don't trip the timeout.
DEFAULT_CONNECT_TIMEOUT = 15.0
DEFAULT_READ_TIMEOUT = 60.0

DEFAULT_RETRY_TOTAL = 5
DEFAULT_RETRY_CONNECT = 5
DEFAULT_RETRY_READ = 5
DEFAULT_RETRY_STATUS = 5
DEFAULT_RETRY_BACKOFF_FACTOR = 0.5
DEFAULT_RETRY_BACKOFF_JITTER = 0.5
DEFAULT_RETRY_BACKOFF_MAX = 30.0
DEFAULT_RETRY_STATUS_FORCELIST = (408, 425, 429, 500, 502, 503, 504)
# Matches urllib3's own ``DEFAULT_ALLOWED_METHODS`` minus TRACE. POST is
# intentionally excluded — a server that processed a POST then returned a
# 5xx would see a duplicate resource on retry. The Memory API's
# ``POST /v1/memory-cards`` creates a new entity, so it must not be
# retried at the transport layer.
DEFAULT_RETRY_ALLOWED_METHODS = (
    "HEAD",
    "GET",
    "OPTIONS",
    "PUT",
    "DELETE",
)

# Methods set for callers whose POST endpoints are idempotent from the
# server's perspective (LLM chat completions are stateless; a duplicate
# request produces a fresh sampled output but no server-side side
# effect, so a transient 429 / 5xx is safe to retry).  Use
# :func:`build_retry_for_idempotent_post` to mint a ``Retry`` policy
# that includes POST.
DEFAULT_RETRY_ALLOWED_METHODS_WITH_POST = (
    "HEAD",
    "GET",
    "OPTIONS",
    "PUT",
    "DELETE",
    "POST",
)


def _normalize_timeout(
    value: float | tuple[float, float],
) -> tuple[float, float]:
    """Normalize a single-float or ``(connect, read)`` tuple to a tuple."""
    if isinstance(value, (int, float)):
        f = float(value)
        return (f, f)
    return (float(value[0]), float(value[1]))


class _TimeoutSession(requests.Session):
    """``requests.Session`` that injects a default per-request timeout.

    Default is a ``(connect, read)`` tuple so each phase is bounded
    independently.  Callers can still pass ``timeout=`` per call (single
    float or tuple).  An explicit ``timeout=None`` is treated the same
    as omitting the kwarg: the session default applies.  ``requests``
    interprets ``timeout=None`` as "wait forever", which is never the
    intent when this session is in use — the whole point of the class
    is to bound every request.
    """

    def __init__(self, *, default_timeout: tuple[float, float]) -> None:
        super().__init__()
        self._default_timeout = default_timeout

    def request(
        self,
        method: str | bytes,
        url: str | bytes,
        *args: Any,
        **kwargs: Any,
    ) -> requests.Response:
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self._default_timeout
        return super().request(method, url, *args, **kwargs)


class _KeepaliveHTTPAdapter(HTTPAdapter):
    """``HTTPAdapter`` that pushes ``socket_options`` and an SSL context
    through to ``urllib3.PoolManager``.

    ``HTTPAdapter`` itself doesn't expose ``socket_options`` or
    ``ssl_context`` as constructor args — both must be threaded through
    ``init_poolmanager`` into the underlying ``PoolManager``. The
    socket-level keepalive options have no effect unless they reach
    every minted connection, which only happens via the pool manager's
    ``connection_pool_kw``.
    """

    def __init__(
        self,
        *args: Any,
        socket_options: list[tuple[int, int, int]] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        **kwargs: Any,
    ) -> None:
        self._socket_options = (
            socket_options
            if socket_options is not None
            else list(DEFAULT_SOCKET_OPTIONS)
        )
        self._ssl_context = (
            ssl_context if ssl_context is not None else build_tls_context()
        )
        super().__init__(*args, **kwargs)

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = False,
        **pool_kwargs: Any,
    ) -> None:
        # urllib3 forwards these to every connection pool, and from
        # there into HTTPConnection / HTTPSConnection.
        pool_kwargs.setdefault("socket_options", self._socket_options)
        pool_kwargs.setdefault("ssl_context", self._ssl_context)
        self._pool_connections = connections
        self._pool_maxsize = maxsize
        self._pool_block = block
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )


def build_retry(
    *,
    total: int = DEFAULT_RETRY_TOTAL,
    connect: int | None = DEFAULT_RETRY_CONNECT,
    read: int | None = DEFAULT_RETRY_READ,
    status: int | None = DEFAULT_RETRY_STATUS,
    backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR,
    backoff_jitter: float = DEFAULT_RETRY_BACKOFF_JITTER,
    backoff_max: float = DEFAULT_RETRY_BACKOFF_MAX,
    status_forcelist: tuple[int, ...] = DEFAULT_RETRY_STATUS_FORCELIST,
    allowed_methods: tuple[str, ...] = DEFAULT_RETRY_ALLOWED_METHODS,
) -> Retry:
    """Build a ``urllib3.Retry`` with the module defaults.

    Separates ``connect`` / ``read`` / ``status`` retry counts so a
    connection problem and a server-error response have independent
    budgets. ``backoff_factor=0.5`` with ``backoff_jitter=0.5`` produces
    randomized exponential sleeps (~0.25–0.75s, ~0.5–1.5s, ~1–3s, …)
    capped at ``backoff_max=30s``, defeating synchronized retry storms.
    ``status_forcelist`` includes ``408`` (Request Timeout, server
    couldn't fully read the request) and ``425`` (Too Early — client
    should resend) along with the standard 429 + 5xx. POST is excluded
    from ``allowed_methods`` because retrying a successfully-processed
    POST creates a duplicate resource. ``raise_on_status=False`` so the
    application formats its own error message from the final response.
    """
    return Retry(
        total=total,
        connect=connect,
        read=read,
        status=status,
        backoff_factor=backoff_factor,
        backoff_jitter=backoff_jitter,
        backoff_max=backoff_max,
        status_forcelist=list(status_forcelist),
        allowed_methods=list(allowed_methods),
        respect_retry_after_header=True,
        raise_on_status=False,
    )


def build_retry_for_idempotent_post(
    *,
    total: int = DEFAULT_RETRY_TOTAL,
    connect: int | None = DEFAULT_RETRY_CONNECT,
    read: int | None = DEFAULT_RETRY_READ,
    status: int | None = DEFAULT_RETRY_STATUS,
    backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR,
    backoff_jitter: float = DEFAULT_RETRY_BACKOFF_JITTER,
    backoff_max: float = DEFAULT_RETRY_BACKOFF_MAX,
    status_forcelist: tuple[int, ...] = DEFAULT_RETRY_STATUS_FORCELIST,
    allowed_methods: tuple[str, ...] = DEFAULT_RETRY_ALLOWED_METHODS_WITH_POST,
) -> Retry:
    """Variant of :func:`build_retry` that includes POST in ``allowed_methods``.

    Use ONLY for callers whose server-side POST is idempotent — every
    LLM chat completion endpoint qualifies (the server holds no state
    about prior requests; the same prompt produces a fresh sampled
    completion but no side effect).  ``POST /v1/memory-cards`` and any
    other resource-creating POST do NOT qualify and must keep the
    conservative :func:`build_retry` policy.

    The retry policy honours ``Retry-After`` on 429 / 503 responses so
    callers respect server-supplied backoff before retrying.  All other
    arguments mirror :func:`build_retry`; ``allowed_methods`` defaults
    to :data:`DEFAULT_RETRY_ALLOWED_METHODS_WITH_POST` but is overridable
    for callers that need a narrower set.
    """
    return Retry(
        total=total,
        connect=connect,
        read=read,
        status=status,
        backoff_factor=backoff_factor,
        backoff_jitter=backoff_jitter,
        backoff_max=backoff_max,
        status_forcelist=list(status_forcelist),
        allowed_methods=list(allowed_methods),
        respect_retry_after_header=True,
        raise_on_status=False,
    )


def make_requests_session(
    role: str,
    *,
    timeout: float | tuple[float, float] = (
        DEFAULT_CONNECT_TIMEOUT,
        DEFAULT_READ_TIMEOUT,
    ),
    pool_connections: int = DEFAULT_POOL_CONNECTIONS,
    pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
    pool_block: bool = False,
    retry: Retry | None = None,
    socket_options: list[tuple[int, int, int]] | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> requests.Session:
    """Create a ``requests.Session`` with the module defaults.

    Args:
        role: Informational label (e.g. ``"memory_api"``). Currently
            unused beyond receipt; reserved for per-role overrides.
        timeout: Default per-request timeout. A single ``float`` applies
            to both connect and read; a ``(connect, read)`` tuple bounds
            each phase. Default ``(15.0, 60.0)``.
        pool_connections: ``HTTPAdapter.pool_connections`` — number of
            host-keyed connection pools to cache. Default ``20``.
        pool_maxsize: ``HTTPAdapter.pool_maxsize`` — maximum connections
            per pool. Default ``100``.
        pool_block: When ``True``, requests block on a full pool rather
            than opening unpooled extra connections. Default ``False``.
        retry: Optional override. When ``None``, :func:`build_retry`
            is used.
        socket_options: Forwarded to every minted connection. Default is
            :data:`gigaevo.infra._net.DEFAULT_SOCKET_OPTIONS` (TCP
            ``SO_KEEPALIVE`` + ``TCP_NODELAY`` + Linux/macOS keepalive
            timing).
        ssl_context: Forwarded to HTTPS connection minting. Default is
            :func:`gigaevo.infra._net.build_tls_context` (TLS 1.2+,
            hostname + CA verification on).

    Returns:
        A ``requests.Session`` with :class:`_KeepaliveHTTPAdapter`
        mounted on both ``http://`` and ``https://``. Caller owns the
        lifetime; call ``session.close()`` when done.
    """
    del role  # informational only
    session = _TimeoutSession(default_timeout=_normalize_timeout(timeout))
    # Identify our traffic in upstream server logs / WAFs / rate-limiters
    # instead of bucketing with anonymous ``python-requests`` traffic.
    session.headers["User-Agent"] = user_agent()
    adapter = _KeepaliveHTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        pool_block=pool_block,
        max_retries=retry if retry is not None else build_retry(),
        socket_options=socket_options,
        ssl_context=ssl_context,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
