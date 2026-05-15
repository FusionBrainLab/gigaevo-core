"""HTTP client factory for long-running asyncio paths.

Centralizes construction of aiohttp.ClientSession / openai DefaultAioHttpClient
with hardened defaults. Two factories are exposed:

- :func:`make_aiohttp_session`  for direct aiohttp consumers (e.g. the Memory
  API client). Caller owns the session lifetime.
- :func:`make_openai_http_client`  for openai-SDK consumers (AsyncOpenAI,
  ChatOpenAI). Returns ``openai.DefaultAioHttpClient`` which speaks the
  httpx-compatible openai SDK interface but routes traffic via aiohttp.

The defaults target the failure class documented in upstream encode/httpx
issue threads referenced from gigaevo-core issue #9: long-running asyncio
processes accumulate connections that httpcore's async semaphore never
releases, producing silent ``PoolTimeout`` cascades on the timescale of
hours. aiohttp's ``TCPConnector(enable_cleanup_closed=True)`` is the
documented defense for the same connection-leak class. Bounded
``ClientTimeout`` (no ``None`` pool waits) prevents the silent forever-hang.

The ``role`` argument on both factories is informational — currently unused
beyond receipt, but exists so per-role overrides (different limits for
memory_api vs llm_calls) can land later without churning call sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    from openai import DefaultAioHttpClient


# ---------------------------------------------------------------------------
# Hardened defaults
# ---------------------------------------------------------------------------

DEFAULT_LIMIT = 300
DEFAULT_LIMIT_PER_HOST = 50
DEFAULT_KEEPALIVE_TIMEOUT = 30.0
DEFAULT_DNS_CACHE_TTL = 300

DEFAULT_TIMEOUT_TOTAL: float | None = 120.0
DEFAULT_TIMEOUT_CONNECT = 10.0
DEFAULT_TIMEOUT_SOCK_READ = 120.0
DEFAULT_TIMEOUT_SOCK_CONNECT = 10.0


def build_connector(
    *,
    limit: int = DEFAULT_LIMIT,
    limit_per_host: int = DEFAULT_LIMIT_PER_HOST,
    keepalive_timeout: float = DEFAULT_KEEPALIVE_TIMEOUT,
    ttl_dns_cache: int = DEFAULT_DNS_CACHE_TTL,
    force_close: bool = False,
) -> aiohttp.TCPConnector:
    """Build a hardened aiohttp.TCPConnector.

    ``enable_cleanup_closed=True`` is always on — it's the upstream-documented
    defense against the connection-leak failure mode and the explicit reason
    to prefer aiohttp's pool over httpcore's async semaphore for long-running
    processes.
    """
    return aiohttp.TCPConnector(
        limit=limit,
        limit_per_host=limit_per_host,
        keepalive_timeout=keepalive_timeout,
        ttl_dns_cache=ttl_dns_cache,
        force_close=force_close,
        enable_cleanup_closed=True,
    )


def build_timeout(
    *,
    total: float | None = DEFAULT_TIMEOUT_TOTAL,
    connect: float | None = DEFAULT_TIMEOUT_CONNECT,
    sock_read: float | None = DEFAULT_TIMEOUT_SOCK_READ,
    sock_connect: float | None = DEFAULT_TIMEOUT_SOCK_CONNECT,
) -> aiohttp.ClientTimeout:
    """Build an aiohttp.ClientTimeout with bounded values.

    Every component is bounded by default. ``None`` is preserved when passed
    explicitly so callers can opt-out per-component (e.g. a streaming
    endpoint that legitimately needs unbounded ``sock_read``), but the
    defaults never produce a session that can hang silently.
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
    """Create an ``aiohttp.ClientSession`` with hardened defaults.

    Args:
        role: Informational label (e.g. ``"memory_api"``, ``"llm_prompts"``).
            Not currently wired to telemetry — reserved for per-role
            override hooks.
        connector: Optional override. When ``None``, :func:`build_connector`
            with hardened defaults is used.
        timeout: Optional override. When ``None``, :func:`build_timeout`
            with hardened defaults is used.
        trust_env: If ``True`` (default), aiohttp honors ``HTTP_PROXY`` /
            ``HTTPS_PROXY`` / ``NO_PROXY`` from the environment. Set
            ``False`` to skip env detection.
        **session_kwargs: Forwarded to ``aiohttp.ClientSession``.

    Returns:
        Open ``aiohttp.ClientSession``. Caller owns the lifetime —
        ``await session.close()`` when done.
    """
    del role  # informational only
    return aiohttp.ClientSession(
        connector=connector if connector is not None else build_connector(),
        timeout=timeout if timeout is not None else build_timeout(),
        trust_env=trust_env,
        **session_kwargs,
    )


def make_openai_http_client(
    role: str,
    *,
    proxy: str | None = None,
    **overrides: Any,
) -> DefaultAioHttpClient:
    """Create an aiohttp-backed http_client for the openai SDK.

    Args:
        role: Informational label (e.g. ``"llm_prompts"``, ``"llm_chains"``).
            Currently unused beyond receipt; reserved for future per-role
            overrides.
        proxy: Optional proxy URL forwarded to ``DefaultAioHttpClient``.
        **overrides: Forwarded to ``DefaultAioHttpClient``.

    Returns:
        ``openai.DefaultAioHttpClient`` instance.

    Raises:
        ImportError: When the ``openai[aiohttp]`` extra is not installed.
            Re-raised with install instructions in the message so callers
            don't have to grok the openai SDK extras layout.
    """
    del role  # informational only
    try:
        from openai import DefaultAioHttpClient
    except ImportError as exc:
        raise ImportError(
            "openai.DefaultAioHttpClient is unavailable. Install the "
            "aiohttp extra: pip install 'openai[aiohttp]>=2.0.0'."
        ) from exc

    kwargs: dict[str, Any] = dict(overrides)
    if proxy is not None:
        kwargs.setdefault("proxy", proxy)
    return DefaultAioHttpClient(**kwargs)
