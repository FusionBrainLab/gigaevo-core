"""Unit tests for :mod:`gigaevo.infra.aiohttp_factory`."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import aiohttp
import pytest

from gigaevo.infra.aiohttp_factory import (
    DEFAULT_KEEPALIVE_TIMEOUT,
    DEFAULT_LIMIT,
    DEFAULT_LIMIT_PER_HOST,
    DEFAULT_TIMEOUT_CONNECT,
    DEFAULT_TIMEOUT_SOCK_CONNECT,
    DEFAULT_TIMEOUT_SOCK_READ,
    DEFAULT_TIMEOUT_TOTAL,
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

    async def test_defaults_match_module_constants(self) -> None:
        c = build_connector()
        try:
            assert c.limit == DEFAULT_LIMIT
            assert c.limit_per_host == DEFAULT_LIMIT_PER_HOST
            assert c._keepalive_timeout == DEFAULT_KEEPALIVE_TIMEOUT
        finally:
            await c.close()

    async def test_enable_cleanup_closed_is_always_passed(self) -> None:
        """Factory unconditionally passes ``enable_cleanup_closed=True``.

        aiohttp 3.10+ on Python 3.12+ internally treats this as a no-op
        (the SSL transport leak it defended against was fixed upstream),
        but the factory still passes it so behavior is correct on older
        Python / older aiohttp where the flag still matters.
        """
        with patch.object(aiohttp, "TCPConnector") as mock_ctor:
            mock_ctor.return_value = mock_ctor  # avoid double-await on close
            build_connector()
        _, kwargs = mock_ctor.call_args
        assert kwargs.get("enable_cleanup_closed") is True

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
# build_timeout
# ---------------------------------------------------------------------------


class TestBuildTimeout:
    def test_defaults_are_bounded(self) -> None:
        """No component defaults to None — the chains/client.py bug surface."""
        t = build_timeout()
        assert t.total == DEFAULT_TIMEOUT_TOTAL
        assert t.connect == DEFAULT_TIMEOUT_CONNECT
        assert t.sock_read == DEFAULT_TIMEOUT_SOCK_READ
        assert t.sock_connect == DEFAULT_TIMEOUT_SOCK_CONNECT
        assert t.total is not None
        assert t.connect is not None
        assert t.sock_read is not None

    def test_explicit_none_passes_through(self) -> None:
        """Streaming endpoints may legitimately need unbounded sock_read."""
        t = build_timeout(sock_read=None)
        assert t.sock_read is None
        assert t.total == DEFAULT_TIMEOUT_TOTAL


# ---------------------------------------------------------------------------
# make_aiohttp_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMakeAiohttpSession:
    async def test_returns_open_session_with_hardened_defaults(self) -> None:
        session = make_aiohttp_session("test_role")
        try:
            assert isinstance(session, aiohttp.ClientSession)
            assert session.closed is False
            assert session._connector is not None
            assert session._connector.limit == DEFAULT_LIMIT
            assert session._timeout.total == DEFAULT_TIMEOUT_TOTAL
        finally:
            await session.close()

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
