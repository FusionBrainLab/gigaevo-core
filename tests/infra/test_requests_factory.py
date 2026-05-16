"""Unit tests for :mod:`gigaevo.infra.requests_factory`."""

from __future__ import annotations

import ssl
from unittest.mock import patch

import requests
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry

from gigaevo.infra._net import DEFAULT_SOCKET_OPTIONS
from gigaevo.infra.requests_factory import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_POOL_CONNECTIONS,
    DEFAULT_POOL_MAXSIZE,
    DEFAULT_READ_TIMEOUT,
    DEFAULT_RETRY_ALLOWED_METHODS,
    DEFAULT_RETRY_BACKOFF_FACTOR,
    DEFAULT_RETRY_BACKOFF_JITTER,
    DEFAULT_RETRY_BACKOFF_MAX,
    DEFAULT_RETRY_CONNECT,
    DEFAULT_RETRY_READ,
    DEFAULT_RETRY_STATUS,
    DEFAULT_RETRY_STATUS_FORCELIST,
    DEFAULT_RETRY_TOTAL,
    _KeepaliveHTTPAdapter,
    _normalize_timeout,
    _TimeoutSession,
    build_retry,
    make_requests_session,
)

# ---------------------------------------------------------------------------
# build_retry
# ---------------------------------------------------------------------------


class TestBuildRetry:
    def test_defaults_match_module_constants(self) -> None:
        r = build_retry()
        assert r.total == DEFAULT_RETRY_TOTAL
        assert r.connect == DEFAULT_RETRY_CONNECT
        assert r.read == DEFAULT_RETRY_READ
        assert r.status == DEFAULT_RETRY_STATUS
        assert r.backoff_factor == DEFAULT_RETRY_BACKOFF_FACTOR
        assert r.backoff_jitter == DEFAULT_RETRY_BACKOFF_JITTER
        assert r.backoff_max == DEFAULT_RETRY_BACKOFF_MAX
        assert r.status_forcelist == list(DEFAULT_RETRY_STATUS_FORCELIST)
        # raise_on_status=False so the Session sees the final response
        # instead of urllib3 turning a retried-but-still-5xx into an
        # exception. The Memory API client formats its own message.
        assert r.raise_on_status is False
        assert r.respect_retry_after_header is True

    def test_post_is_not_in_default_allowed_methods(self) -> None:
        """Retrying POST after a 5xx can duplicate a successfully-processed
        create. urllib3's safe default excludes POST; the factory matches."""
        assert "POST" not in DEFAULT_RETRY_ALLOWED_METHODS
        r = build_retry()
        assert "POST" not in r.allowed_methods

    def test_status_forcelist_includes_429_and_408(self) -> None:
        """429 (rate-limited, honors Retry-After) and 408 (server-side
        request timeout) belong on the retry list along with 5xx."""
        assert 429 in DEFAULT_RETRY_STATUS_FORCELIST
        assert 408 in DEFAULT_RETRY_STATUS_FORCELIST
        assert 503 in DEFAULT_RETRY_STATUS_FORCELIST

    def test_overrides_applied(self) -> None:
        r = build_retry(
            total=5,
            connect=2,
            backoff_factor=2.0,
            status_forcelist=(429, 503),
            allowed_methods=("GET",),
        )
        assert r.total == 5
        assert r.connect == 2
        assert r.backoff_factor == 2.0
        assert r.status_forcelist == [429, 503]


# ---------------------------------------------------------------------------
# _normalize_timeout
# ---------------------------------------------------------------------------


class TestNormalizeTimeout:
    def test_single_float_applies_to_both_phases(self) -> None:
        assert _normalize_timeout(15.0) == (15.0, 15.0)

    def test_int_is_promoted_to_float(self) -> None:
        assert _normalize_timeout(15) == (15.0, 15.0)

    def test_tuple_passes_through(self) -> None:
        assert _normalize_timeout((5.0, 60.0)) == (5.0, 60.0)


# ---------------------------------------------------------------------------
# _TimeoutSession
# ---------------------------------------------------------------------------


class TestTimeoutSession:
    def test_default_timeout_injected(self) -> None:
        session = _TimeoutSession(default_timeout=(5.0, 7.5))
        with patch.object(requests.Session, "request", autospec=True) as mock_super:
            mock_super.return_value = "ok"
            session.request("GET", "http://example.com")
        assert mock_super.call_args.kwargs["timeout"] == (5.0, 7.5)

    def test_caller_timeout_wins(self) -> None:
        session = _TimeoutSession(default_timeout=(5.0, 7.5))
        with patch.object(requests.Session, "request", autospec=True) as mock_super:
            mock_super.return_value = "ok"
            session.request("GET", "http://example.com", timeout=1.0)
        assert mock_super.call_args.kwargs["timeout"] == 1.0

    def test_explicit_none_timeout_treated_as_default(self) -> None:
        """``requests`` interprets ``timeout=None`` as "wait forever".
        This session is supposed to bound every request, so an explicit
        ``None`` must still hit the session default."""
        session = _TimeoutSession(default_timeout=(5.0, 7.5))
        with patch.object(requests.Session, "request", autospec=True) as mock_super:
            mock_super.return_value = "ok"
            session.request("GET", "http://example.com", timeout=None)
        assert mock_super.call_args.kwargs["timeout"] == (5.0, 7.5)

    def test_extra_kwargs_forwarded_verbatim(self) -> None:
        """Subclass must forward params= / json= / headers= unchanged."""
        session = _TimeoutSession(default_timeout=(5.0, 7.5))
        with patch.object(requests.Session, "request", autospec=True) as mock_super:
            mock_super.return_value = "ok"
            session.request(
                "POST",
                "http://example.com",
                json={"x": 1},
                params={"a": "b"},
                headers={"h": "v"},
            )
        kwargs = mock_super.call_args.kwargs
        assert kwargs["json"] == {"x": 1}
        assert kwargs["params"] == {"a": "b"}
        assert kwargs["headers"] == {"h": "v"}
        assert kwargs["timeout"] == (5.0, 7.5)


# ---------------------------------------------------------------------------
# _KeepaliveHTTPAdapter
# ---------------------------------------------------------------------------


class TestKeepaliveHTTPAdapter:
    def test_socket_options_default_to_module_constant(self) -> None:
        adapter = _KeepaliveHTTPAdapter()
        try:
            assert adapter._socket_options == list(DEFAULT_SOCKET_OPTIONS)
        finally:
            adapter.close()

    def test_ssl_context_default_pins_tls12_and_verifies(self) -> None:
        adapter = _KeepaliveHTTPAdapter()
        try:
            assert isinstance(adapter._ssl_context, ssl.SSLContext)
            assert adapter._ssl_context.minimum_version == ssl.TLSVersion.TLSv1_2
            assert adapter._ssl_context.check_hostname is True
            assert adapter._ssl_context.verify_mode == ssl.CERT_REQUIRED
        finally:
            adapter.close()

    def test_init_poolmanager_threads_socket_options_through(self) -> None:
        """``socket_options`` must reach ``PoolManager`` so every minted
        connection inherits the keepalive settings — not just the first."""
        adapter = _KeepaliveHTTPAdapter(socket_options=[(1, 2, 3)])
        try:
            assert isinstance(adapter.poolmanager, PoolManager)
            assert adapter.poolmanager.connection_pool_kw.get(
                "socket_options"
            ) == [(1, 2, 3)]
        finally:
            adapter.close()

    def test_init_poolmanager_threads_ssl_context_through(self) -> None:
        custom_ctx = ssl.create_default_context()
        adapter = _KeepaliveHTTPAdapter(ssl_context=custom_ctx)
        try:
            assert adapter.poolmanager.connection_pool_kw.get("ssl_context") is custom_ctx
        finally:
            adapter.close()


# ---------------------------------------------------------------------------
# make_requests_session
# ---------------------------------------------------------------------------


class TestMakeRequestsSession:
    def test_returns_session_with_module_defaults(self) -> None:
        session = make_requests_session("test_role")
        try:
            assert isinstance(session, requests.Session)
            adapter_http = session.get_adapter("http://x")
            adapter_https = session.get_adapter("https://x")
            assert isinstance(adapter_http, _KeepaliveHTTPAdapter)
            assert isinstance(adapter_https, _KeepaliveHTTPAdapter)
            assert adapter_http._pool_connections == DEFAULT_POOL_CONNECTIONS
            assert adapter_http._pool_maxsize == DEFAULT_POOL_MAXSIZE
            assert isinstance(adapter_http.max_retries, Retry)
            assert adapter_http.max_retries.total == DEFAULT_RETRY_TOTAL
        finally:
            session.close()

    def test_default_timeout_split_into_tuple(self) -> None:
        session = make_requests_session("test_role")
        try:
            assert isinstance(session, _TimeoutSession)
            assert session._default_timeout == (
                DEFAULT_CONNECT_TIMEOUT,
                DEFAULT_READ_TIMEOUT,
            )
        finally:
            session.close()

    def test_single_float_timeout_applies_to_both(self) -> None:
        session = make_requests_session("test_role", timeout=12.0)
        try:
            assert isinstance(session, _TimeoutSession)
            assert session._default_timeout == (12.0, 12.0)
        finally:
            session.close()

    def test_tuple_timeout_used_as_is(self) -> None:
        session = make_requests_session("test_role", timeout=(2.0, 99.0))
        try:
            assert session._default_timeout == (2.0, 99.0)
        finally:
            session.close()

    def test_custom_pool_sizes(self) -> None:
        session = make_requests_session(
            "test_role", pool_connections=4, pool_maxsize=8
        )
        try:
            adapter = session.get_adapter("https://x")
            assert isinstance(adapter, _KeepaliveHTTPAdapter)
            assert adapter._pool_connections == 4
            assert adapter._pool_maxsize == 8
        finally:
            session.close()

    def test_custom_retry(self) -> None:
        custom = build_retry(total=99)
        session = make_requests_session("test_role", retry=custom)
        try:
            adapter = session.get_adapter("https://x")
            assert isinstance(adapter, _KeepaliveHTTPAdapter)
            assert adapter.max_retries is custom
            assert adapter.max_retries.total == 99
        finally:
            session.close()

    def test_same_adapter_mounted_on_http_and_https(self) -> None:
        """Both prefixes share the same adapter instance so the pool is
        unified across schemes."""
        session = make_requests_session("test_role")
        try:
            assert session.get_adapter("http://x") is session.get_adapter("https://x")
        finally:
            session.close()

    def test_socket_options_reach_pool_manager(self) -> None:
        """The socket-options-on-every-connection guarantee depends on the
        adapter pushing them through to the PoolManager."""
        session = make_requests_session("test_role")
        try:
            adapter = session.get_adapter("https://x")
            assert isinstance(adapter, _KeepaliveHTTPAdapter)
            pool_socket_options = adapter.poolmanager.connection_pool_kw.get(
                "socket_options"
            )
            assert pool_socket_options == list(DEFAULT_SOCKET_OPTIONS)
        finally:
            session.close()

    def test_user_agent_header_set(self) -> None:
        """Identifies our traffic in upstream server logs instead of
        bucketing with anonymous ``python-requests`` traffic."""
        session = make_requests_session("test_role")
        try:
            ua = session.headers.get("User-Agent", "")
            assert ua.startswith("gigaevo-core/")
        finally:
            session.close()
