"""Hardened ``requests.Session`` factory for sync HTTP paths.

The sync counterpart to :mod:`gigaevo.infra.aiohttp_factory`. Builds a
``requests.Session`` whose underlying ``urllib3`` connection pool is
explicitly sized and whose retry policy is wired through
``urllib3.util.retry.Retry`` — retries happen inside ``urllib3`` before
the response reaches application code, which is a meaningful
reliability property versus tenacity-style wrappers.

A thin ``Session`` subclass injects a default per-request timeout, since
``requests`` has no session-level timeout knob. Callers that want a
different timeout for a specific call can still pass ``timeout=`` per
request — the subclass only applies the default when none is given.
"""

from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Hardened defaults
# ---------------------------------------------------------------------------

DEFAULT_POOL_CONNECTIONS = 10
DEFAULT_POOL_MAXSIZE = 50
DEFAULT_TIMEOUT = 30.0

DEFAULT_RETRY_TOTAL = 3
DEFAULT_RETRY_BACKOFF_FACTOR = 0.5
DEFAULT_RETRY_STATUS_FORCELIST = (500, 502, 503, 504)
DEFAULT_RETRY_ALLOWED_METHODS = (
    "HEAD",
    "GET",
    "OPTIONS",
    "POST",
    "PUT",
    "DELETE",
)


class _TimeoutSession(requests.Session):
    """``requests.Session`` that injects a default per-request timeout.

    ``requests`` has no session-level timeout — every ``.request`` call
    must pass one or hang on read indefinitely. This subclass sets the
    default but lets callers override per-call.
    """

    def __init__(self, *, default_timeout: float = DEFAULT_TIMEOUT) -> None:
        super().__init__()
        self._default_timeout = default_timeout

    def request(self, method: str | bytes, url: str | bytes, **kwargs: Any) -> requests.Response:  # type: ignore[override]
        kwargs.setdefault("timeout", self._default_timeout)
        return super().request(method, url, **kwargs)


def build_retry(
    *,
    total: int = DEFAULT_RETRY_TOTAL,
    backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR,
    status_forcelist: tuple[int, ...] = DEFAULT_RETRY_STATUS_FORCELIST,
    allowed_methods: tuple[str, ...] = DEFAULT_RETRY_ALLOWED_METHODS,
) -> Retry:
    """Build a ``urllib3.Retry`` with hardened defaults.

    ``backoff_factor=0.5`` produces 0.5s / 1s / 2s / 4s sleeps between
    retries. ``status_forcelist`` triggers retry for transient 5xx
    responses. ``allowed_methods`` permits retry on the methods the
    Memory API uses, including POST/PUT/DELETE — Memory API entity
    operations are idempotent (entity_id-keyed), so retry on these is
    safe and matches the legacy behavior.
    """
    return Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods=list(allowed_methods),
        raise_on_status=False,
    )


def make_requests_session(
    role: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    pool_connections: int = DEFAULT_POOL_CONNECTIONS,
    pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
    retry: Retry | None = None,
) -> requests.Session:
    """Create a hardened ``requests.Session``.

    Args:
        role: Informational label (e.g. ``"memory_api"``). Currently
            unused beyond receipt; reserved for per-role overrides.
        timeout: Default per-request timeout, applied when the caller
            does not pass ``timeout=`` explicitly. Default ``30.0``s.
        pool_connections: ``HTTPAdapter.pool_connections`` — number of
            connection pools to cache (one per host).
        pool_maxsize: ``HTTPAdapter.pool_maxsize`` — maximum connections
            per pool.
        retry: Optional override. When ``None``, :func:`build_retry`
            with hardened defaults is used.

    Returns:
        A ``requests.Session`` with the HTTPAdapter mounted on both
        ``http://`` and ``https://`` prefixes. Caller owns the lifetime;
        call ``session.close()`` when done.
    """
    del role  # informational only
    session = _TimeoutSession(default_timeout=timeout)
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=retry if retry is not None else build_retry(),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
