"""Unit tests for :mod:`gigaevo.infra.requests_factory`."""

from __future__ import annotations

from unittest.mock import patch

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gigaevo.infra.requests_factory import (
    DEFAULT_POOL_CONNECTIONS,
    DEFAULT_POOL_MAXSIZE,
    DEFAULT_RETRY_BACKOFF_FACTOR,
    DEFAULT_RETRY_STATUS_FORCELIST,
    DEFAULT_RETRY_TOTAL,
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
        assert r.backoff_factor == DEFAULT_RETRY_BACKOFF_FACTOR
        assert r.status_forcelist == list(DEFAULT_RETRY_STATUS_FORCELIST)
        # raise_on_status=False so the Session can see the final response
        # instead of urllib3 turning a retried-but-still-5xx into an
        # exception. The Memory API client wants to format its own message.
        assert r.raise_on_status is False

    def test_overrides_applied(self) -> None:
        r = build_retry(
            total=5,
            backoff_factor=2.0,
            status_forcelist=(429, 503),
            allowed_methods=("GET",),
        )
        assert r.total == 5
        assert r.backoff_factor == 2.0
        assert r.status_forcelist == [429, 503]


# ---------------------------------------------------------------------------
# _TimeoutSession
# ---------------------------------------------------------------------------


class TestTimeoutSession:
    def test_default_timeout_injected(self) -> None:
        session = _TimeoutSession(default_timeout=7.5)
        captured: dict = {}

        with patch.object(requests.Session, "request", autospec=True) as mock_super:
            mock_super.return_value = "ok"
            session.request("GET", "http://example.com")
            args, kwargs = mock_super.call_args
            captured.update(kwargs)
        assert captured["timeout"] == 7.5

    def test_caller_timeout_wins(self) -> None:
        session = _TimeoutSession(default_timeout=7.5)
        captured: dict = {}

        with patch.object(requests.Session, "request", autospec=True) as mock_super:
            mock_super.return_value = "ok"
            session.request("GET", "http://example.com", timeout=1.0)
            args, kwargs = mock_super.call_args
            captured.update(kwargs)
        assert captured["timeout"] == 1.0


# ---------------------------------------------------------------------------
# make_requests_session
# ---------------------------------------------------------------------------


class TestMakeRequestsSession:
    def test_returns_session_with_hardened_defaults(self) -> None:
        session = make_requests_session("test_role")
        try:
            assert isinstance(session, requests.Session)
            adapter_http = session.get_adapter("http://x")
            adapter_https = session.get_adapter("https://x")
            assert isinstance(adapter_http, HTTPAdapter)
            assert isinstance(adapter_https, HTTPAdapter)
            # HTTPAdapter exposes pool-config via attributes set on init
            assert adapter_http._pool_connections == DEFAULT_POOL_CONNECTIONS
            assert adapter_http._pool_maxsize == DEFAULT_POOL_MAXSIZE
            assert isinstance(adapter_http.max_retries, Retry)
            assert adapter_http.max_retries.total == DEFAULT_RETRY_TOTAL
        finally:
            session.close()

    def test_default_timeout_applied_per_request(self) -> None:
        session = make_requests_session("test_role", timeout=12.0)
        try:
            assert isinstance(session, _TimeoutSession)
            assert session._default_timeout == 12.0
        finally:
            session.close()

    def test_custom_pool_sizes(self) -> None:
        session = make_requests_session(
            "test_role", pool_connections=4, pool_maxsize=8
        )
        try:
            adapter = session.get_adapter("https://x")
            assert isinstance(adapter, HTTPAdapter)
            assert adapter._pool_connections == 4
            assert adapter._pool_maxsize == 8
        finally:
            session.close()

    def test_custom_retry(self) -> None:
        custom = build_retry(total=99)
        session = make_requests_session("test_role", retry=custom)
        try:
            adapter = session.get_adapter("https://x")
            assert isinstance(adapter, HTTPAdapter)
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
