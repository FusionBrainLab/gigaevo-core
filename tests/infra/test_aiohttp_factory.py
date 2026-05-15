"""Unit tests for :mod:`gigaevo.infra.aiohttp_factory`."""

from __future__ import annotations

import socket
import ssl
import sys
import types
from unittest.mock import patch

import aiohttp
import pytest

from gigaevo.infra._net import DEFAULT_SOCKET_OPTIONS, apply_socket_options
from gigaevo.infra.aiohttp_factory import (
    DEFAULT_DNS_CACHE_TTL,
    DEFAULT_KEEPALIVE_TIMEOUT,
    DEFAULT_LIMIT,
    DEFAULT_LIMIT_PER_HOST,
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_SOCK_CONNECT,
    DEFAULT_TIMEOUT_SOCK_READ,
    DEFAULT_TIMEOUT_TOTAL,
    _KeepaliveConnector,
    build_connector,
    build_timeout,
    make_aiohttp_session,
    make_openai_http_client,
)

# ---------------------------------------------------------------------------
# build_connector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBuildConnector:
    """``aiohttp.TCPConnector`` requires a running event loop since aiohttp
    3.10+; every test in this class must be async."""

    async def test_returns_keepalive_subclass(self) -> None:
        c = build_connector()
        try:
            assert isinstance(c, _KeepaliveConnector)
            assert isinstance(c, aiohttp.TCPConnector)
        finally:
            await c.close()

    async def test_defaults_match_module_constants(self) -> None:
        c = build_connector()
        try:
            assert c.limit == DEFAULT_LIMIT
            assert c.limit_per_host == DEFAULT_LIMIT_PER_HOST
            assert c._keepalive_timeout == DEFAULT_KEEPALIVE_TIMEOUT
        finally:
            await c.close()

    async def test_keepalive_timeout_beats_common_lb_timeouts(self) -> None:
        """``keepalive_timeout`` must be < the most aggressive common LB
        idle timeout (AWS ALB 60s, NGINX 75s, CloudFlare 90s) so the LB
        never closes a kept-alive socket out from under us."""
        c = build_connector()
        try:
            assert c._keepalive_timeout <= 30.0
        finally:
            await c.close()

    async def test_enable_cleanup_closed_is_always_passed(self) -> None:
        """Factory unconditionally passes ``enable_cleanup_closed=True`` —
        no-op on Py3.12+, helpful on older Python."""
        from gigaevo.infra import aiohttp_factory as factory_mod

        with patch.object(factory_mod, "_KeepaliveConnector") as mock_ctor:
            mock_ctor.return_value = mock_ctor  # avoid double-await on close
            build_connector()
        assert mock_ctor.call_args.kwargs.get("enable_cleanup_closed") is True

    async def test_ssl_context_default_pins_tls12_and_verifies(self) -> None:
        c = build_connector()
        try:
            ctx = c._ssl
            assert isinstance(ctx, ssl.SSLContext)
            assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2
            assert ctx.check_hostname is True
            assert ctx.verify_mode == ssl.CERT_REQUIRED
        finally:
            await c.close()

    async def test_ssl_context_override(self) -> None:
        custom = ssl.create_default_context()
        custom.minimum_version = ssl.TLSVersion.TLSv1_3
        c = build_connector(ssl_context=custom)
        try:
            assert c._ssl is custom
        finally:
            await c.close()

    async def test_overrides_applied(self) -> None:
        c = build_connector(
            limit=42, limit_per_host=7, keepalive_timeout=99.0, ttl_dns_cache=5
        )
        try:
            assert c.limit == 42
            assert c.limit_per_host == 7
            assert c._keepalive_timeout == 99.0
        finally:
            await c.close()


# ---------------------------------------------------------------------------
# Default socket options (shared with requests stack via _net)
# ---------------------------------------------------------------------------


class TestDefaultSocketOptions:
    def test_so_keepalive_is_always_present(self) -> None:
        """The most-portable keepalive option must be there on every OS."""
        assert (
            socket.SOL_SOCKET,
            socket.SO_KEEPALIVE,
            1,
        ) in DEFAULT_SOCKET_OPTIONS

    def test_tcp_nodelay_is_always_present(self) -> None:
        """Replicates urllib3's default so TCP_NODELAY isn't lost when
        callers compose their own option list."""
        assert (
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        ) in DEFAULT_SOCKET_OPTIONS

    @pytest.mark.skipif(
        not hasattr(socket, "TCP_KEEPIDLE"),
        reason="TCP_KEEPIDLE not available on this OS",
    )
    def test_tcp_keepidle_on_linux(self) -> None:
        assert any(
            opt == (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
            for opt in DEFAULT_SOCKET_OPTIONS
        )

    def test_apply_socket_options_swallows_oserror(self) -> None:
        """A bad setsockopt on one option must not bring down the rest."""

        class _StubSock:
            def __init__(self) -> None:
                self.calls: list[tuple[int, int, int]] = []
                self.fail_next = True

            def setsockopt(self, level: int, opt: int, val: int) -> None:
                if self.fail_next:
                    self.fail_next = False
                    raise OSError("kernel rejected")
                self.calls.append((level, opt, val))

        stub = _StubSock()
        apply_socket_options(stub)  # type: ignore[arg-type]
        # First option raised; remaining should have been applied.
        assert len(stub.calls) == len(DEFAULT_SOCKET_OPTIONS) - 1


# ---------------------------------------------------------------------------
# build_timeout
# ---------------------------------------------------------------------------


class TestBuildTimeout:
    def test_defaults_are_bounded(self) -> None:
        """No component defaults to None — silent forever-hang is
        structurally unreachable."""
        t = build_timeout()
        assert t.total == DEFAULT_TIMEOUT_TOTAL
        assert t.connect == DEFAULT_TIMEOUT_CONNECT
        assert t.sock_read == DEFAULT_TIMEOUT_SOCK_READ
        assert t.sock_connect == DEFAULT_TIMEOUT_SOCK_CONNECT
        assert t.total is not None
        assert t.connect is not None
        assert t.sock_read is not None

    def test_explicit_none_passes_through(self) -> None:
        """Streaming endpoints may need unbounded sock_read."""
        t = build_timeout(sock_read=None)
        assert t.sock_read is None
        assert t.total == DEFAULT_TIMEOUT_TOTAL

    def test_llm_friendly_defaults(self) -> None:
        """LLM generations can take several minutes. Defaults must clear
        a generous bar; 60s would be too tight."""
        assert DEFAULT_TIMEOUT_TOTAL is not None
        assert DEFAULT_TIMEOUT_TOTAL >= 180.0
        assert DEFAULT_TIMEOUT_SOCK_READ >= 180.0


# ---------------------------------------------------------------------------
# make_aiohttp_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMakeAiohttpSession:
    async def test_returns_open_session_with_module_defaults(self) -> None:
        session = make_aiohttp_session("test_role")
        try:
            assert isinstance(session, aiohttp.ClientSession)
            assert session.closed is False
            assert session._connector is not None
            assert session._connector.limit == DEFAULT_LIMIT
            assert session._timeout.total == DEFAULT_TIMEOUT_TOTAL
            assert isinstance(session._connector, _KeepaliveConnector)
        finally:
            await session.close()

    async def test_default_dns_cache_ttl_forwarded_to_connector(self) -> None:
        """aiohttp 3.13 doesn't expose ``ttl_dns_cache`` as a public attribute
        on the connector — assert via the construction kwarg instead."""
        from gigaevo.infra import aiohttp_factory as factory_mod

        with patch.object(factory_mod, "_KeepaliveConnector") as mock_ctor:
            mock_ctor.return_value = mock_ctor
            build_connector()
        assert (
            mock_ctor.call_args.kwargs.get("ttl_dns_cache")
            == DEFAULT_DNS_CACHE_TTL
        )

    async def test_trust_env_default_true(self) -> None:
        session = make_aiohttp_session("test_role")
        try:
            assert session.trust_env is True
        finally:
            await session.close()

    async def test_trust_env_override(self) -> None:
        session = make_aiohttp_session("test_role", trust_env=False)
        try:
            assert session.trust_env is False
        finally:
            await session.close()

    async def test_custom_connector_used(self) -> None:
        custom = build_connector(limit=11)
        session = make_aiohttp_session("test_role", connector=custom)
        try:
            assert session._connector is custom
            assert session._connector.limit == 11
        finally:
            await session.close()

    async def test_custom_timeout_used(self) -> None:
        custom = build_timeout(total=5.0)
        session = make_aiohttp_session("test_role", timeout=custom)
        try:
            assert session._timeout is custom
            assert session._timeout.total == 5.0
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# make_openai_http_client
# ---------------------------------------------------------------------------


class TestMakeOpenAIHttpClient:
    def test_constructs_with_role(self) -> None:
        """Verify the factory delegates to openai.DefaultAioHttpClient.

        We stub the openai module so the test does not require the
        ``openai[aiohttp]`` extra at test time.
        """
        constructed: dict[str, object] = {}

        class _FakeDefaultAioHttpClient:
            def __init__(self, **kwargs: object) -> None:
                constructed.update(kwargs)
                self.kwargs = kwargs

        fake_openai = types.SimpleNamespace(
            DefaultAioHttpClient=_FakeDefaultAioHttpClient
        )
        with patch.dict(sys.modules, {"openai": fake_openai}):
            client = make_openai_http_client("llm_role")
        assert isinstance(client, _FakeDefaultAioHttpClient)
        assert constructed == {}

    def test_proxy_forwarded(self) -> None:
        constructed: dict[str, object] = {}

        class _FakeDefaultAioHttpClient:
            def __init__(self, **kwargs: object) -> None:
                constructed.update(kwargs)

        fake_openai = types.SimpleNamespace(
            DefaultAioHttpClient=_FakeDefaultAioHttpClient
        )
        with patch.dict(sys.modules, {"openai": fake_openai}):
            make_openai_http_client("llm_role", proxy="http://proxy:8080")
        assert constructed == {"proxy": "http://proxy:8080"}

    def test_overrides_forwarded(self) -> None:
        constructed: dict[str, object] = {}

        class _FakeDefaultAioHttpClient:
            def __init__(self, **kwargs: object) -> None:
                constructed.update(kwargs)

        fake_openai = types.SimpleNamespace(
            DefaultAioHttpClient=_FakeDefaultAioHttpClient
        )
        with patch.dict(sys.modules, {"openai": fake_openai}):
            make_openai_http_client("llm_role", timeout=42.0, custom="value")
        assert constructed == {"timeout": 42.0, "custom": "value"}

    def test_import_error_surfaces_install_instructions(self) -> None:
        # Force openai import to fail
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(ImportError, match="openai\\[aiohttp\\]"):
                make_openai_http_client("llm_role")
