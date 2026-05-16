"""Unit tests for :mod:`gigaevo.infra._net` helpers shared by both
the aiohttp and requests factories."""

from __future__ import annotations

import ssl

from gigaevo.infra._net import build_tls_context, user_agent


class TestBuildTlsContext:
    def test_pins_tls12_with_verification(self) -> None:
        ctx = build_tls_context()
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2
        assert ctx.check_hostname is True
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_returns_cached_singleton(self) -> None:
        """``SSLContext`` construction parses the CA bundle — non-trivial
        cost.  Cached singleton avoids rebuilding it per session."""
        a = build_tls_context()
        b = build_tls_context()
        assert a is b


class TestUserAgent:
    def test_starts_with_package_prefix(self) -> None:
        ua = user_agent()
        assert ua.startswith("gigaevo-core/")

    def test_returns_cached_value(self) -> None:
        """Memoised — the importlib metadata lookup is cheap but
        unnecessary on the hot path."""
        assert user_agent() is user_agent()
