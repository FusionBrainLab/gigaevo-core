"""Shared networking primitives for the aiohttp + requests factories.

Centralizes the parts that must agree across the sync and async stacks:

- TCP-level dead-peer detection via ``SO_KEEPALIVE`` plus per-OS probe
  timing (``TCP_KEEPIDLE`` / ``TCP_KEEPINTVL`` / ``TCP_KEEPCNT`` on
  Linux; ``TCP_KEEPALIVE`` on macOS). Linux defaults idle-before-probe
  to ``net.ipv4.tcp_keepalive_time=7200`` (two hours), which is useless
  for HTTP. The values below drop that to ``IDLE=60s, INTVL=10s,
  CNT=6`` — a dead peer is detected in roughly 120 seconds rather than
  two hours, while idle-but-alive connections cost a handful of probe
  packets per minute.
- ``TCP_NODELAY=1`` to skip Nagle's algorithm — the default urllib3
  socket option, replicated so callers that compose their own option
  list don't accidentally drop it.
- :func:`build_tls_context` — an ``ssl.SSLContext`` pinned to TLS 1.2+
  with hostname checking and CA verification on. urllib3 / aiohttp
  already default to a similar context; this function makes the
  properties explicit so behavior survives library upgrades.

The constants are module-level so they're computed once at import.
``hasattr`` gates protect against platforms (macOS, Windows) that lack
the Linux-specific options — the option is simply omitted there.
"""

from __future__ import annotations

import functools
import socket
import ssl

# ---------------------------------------------------------------------------
# Default socket options shared by aiohttp + requests
# ---------------------------------------------------------------------------

TCP_KEEPIDLE_SECONDS = 60
TCP_KEEPINTVL_SECONDS = 10
TCP_KEEPCNT = 6


def _build_socket_options() -> list[tuple[int, int, int]]:
    """Build the keepalive socket option list, gated by platform availability."""
    opts: list[tuple[int, int, int]] = [
        # urllib3's default — disable Nagle, latency-friendly for small HTTP
        # payloads. Replicated here so callers replacing socket_options keep it.
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
        # OS-level dead-peer detection — always portable.
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ]
    # Linux per-connection keepalive timing. macOS exposes TCP_KEEPALIVE
    # (different semantics — sets idle time directly). Windows has none of
    # these; SO_KEEPALIVE on Windows uses the system defaults from
    # HKLM\System\CurrentControlSet\Services\Tcpip\Parameters.
    if hasattr(socket, "TCP_KEEPIDLE"):
        opts.append(
            (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, TCP_KEEPIDLE_SECONDS)
        )
    elif hasattr(socket, "TCP_KEEPALIVE"):
        # macOS analogue of TCP_KEEPIDLE (Pyright doesn't see this on Linux).
        opts.append(
            (
                socket.IPPROTO_TCP,
                socket.TCP_KEEPALIVE,  # type: ignore[attr-defined]
                TCP_KEEPIDLE_SECONDS,
            )
        )
    if hasattr(socket, "TCP_KEEPINTVL"):
        opts.append(
            (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, TCP_KEEPINTVL_SECONDS)
        )
    if hasattr(socket, "TCP_KEEPCNT"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, TCP_KEEPCNT))
    return opts


DEFAULT_SOCKET_OPTIONS: list[tuple[int, int, int]] = _build_socket_options()


def apply_socket_options(sock: socket.socket) -> None:
    """Apply :data:`DEFAULT_SOCKET_OPTIONS` to a connected socket.

    Best-effort: each option is set independently and an ``OSError`` on
    an individual setsockopt is swallowed. The aiohttp connector cannot
    afford to fail an established connection because the kernel rejected
    a tuning option.
    """
    for level, optname, value in DEFAULT_SOCKET_OPTIONS:
        try:
            sock.setsockopt(level, optname, value)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# TLS context
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def user_agent() -> str:
    """Return the ``User-Agent`` string used by every outbound HTTP call.

    Format: ``gigaevo-core/<version>``.  Identifies our traffic in
    upstream logs so rate-limiters / WAFs can apply per-client policies
    instead of bucketing us with anonymous ``python-requests`` traffic.
    Falls back to ``"unknown"`` for the version segment when the package
    metadata is unavailable (development checkout without ``pip install``).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            v = version("gigaevo")
        except PackageNotFoundError:
            v = "unknown"
    except ImportError:  # pragma: no cover — stdlib on 3.8+
        v = "unknown"
    return f"gigaevo-core/{v}"


@functools.lru_cache(maxsize=1)
def build_tls_context() -> ssl.SSLContext:
    """Build an ``ssl.SSLContext`` pinned to TLS 1.2+ with verification on.

    Python's ``ssl.create_default_context()`` already produces a TLS
    1.2+ client context with hostname checking and CA verification on,
    sourcing trust from ``certifi`` when installed. This function pins
    those properties explicitly so they survive Python/openssl upgrades
    and are visible in our code rather than implicit in library
    internals.

    The result is cached: ``SSLContext`` is expensive to construct
    (parses the CA bundle) and the returned object is treated as
    read-only by every call site in this package.  Do not mutate the
    returned context; callers needing a customised context should build
    their own via ``ssl.create_default_context``.
    """
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx
