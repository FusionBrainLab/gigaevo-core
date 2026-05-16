"""Unit tests for :mod:`gigaevo.memory.shared_memory.concept_api`.

Focuses on error-path hardening: exception messages must preserve the
underlying transport-level cause without splicing raw response bodies
straight into log sinks.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.shared_memory.concept_api import (
    _ConceptApiClient,
    _safe_response_preview,
)


class TestSafeResponsePreview:
    def _make_response(self, text: str) -> requests.Response:
        r = requests.Response()
        r._content = text.encode("utf-8")
        r.encoding = "utf-8"
        return r

    def test_passes_through_short_clean_body(self) -> None:
        r = self._make_response("plain ok response")
        assert _safe_response_preview(r) == "plain ok response"

    def test_truncates_long_body(self) -> None:
        body = "x" * 4096
        out = _safe_response_preview(self._make_response(body))
        assert out.endswith("… (truncated)")
        assert len(out) < 1024

    def test_strips_ansi_escape_sequences(self) -> None:
        # Classic 'red' ANSI prefix + reset; must not survive into the log.
        body = "\x1b[31merror\x1b[0m"
        out = _safe_response_preview(self._make_response(body))
        assert "\x1b" not in out
        assert "error" in out

    def test_strips_crlf_and_nul(self) -> None:
        body = "line1\r\nline2\x00with-nul"
        out = _safe_response_preview(self._make_response(body))
        assert "\r" not in out
        assert "\x00" not in out
        assert "line2with-nul" in out
        # \n preserved (multi-line responses are common and readable).
        assert "\n" in out


class TestRequestErrorMessages:
    def test_connection_error_preserves_cause(self) -> None:
        """Transport-level diagnostic (DNS / ECONNREFUSED / SSL) must end
        up in the exception message, not just chained via ``__cause__``."""
        client = _ConceptApiClient(base_url="http://nowhere.invalid")
        with patch.object(
            client._http,
            "request",
            side_effect=requests.exceptions.ConnectionError("name resolution failed"),
        ):
            with pytest.raises(MemoryStorageError) as exc_info:
                client._request("GET", "/v1/memory-cards")
        assert "name resolution failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_timeout_preserves_cause(self) -> None:
        client = _ConceptApiClient(base_url="http://nowhere.invalid")
        with patch.object(
            client._http,
            "request",
            side_effect=requests.exceptions.Timeout("read timed out after 60s"),
        ):
            with pytest.raises(MemoryStorageError) as exc_info:
                client._request("GET", "/v1/memory-cards")
        assert "read timed out after 60s" in str(exc_info.value)

    def test_http_error_body_is_truncated_and_sanitised(self) -> None:
        """A 500 with ANSI / huge body must not splice raw bytes into the
        exception message."""
        client = _ConceptApiClient(base_url="http://memory.local")

        response = requests.Response()
        response.status_code = 500
        # Body that mixes ANSI escapes, CR/LF, NUL, and huge length.
        nasty = "\x1b[31mboom\x1b[0m\r\n\x00" + "x" * 4096
        response._content = nasty.encode("utf-8")
        response.encoding = "utf-8"

        with patch.object(client._http, "request", return_value=response):
            with pytest.raises(MemoryStorageError) as exc_info:
                client._request("GET", "/v1/memory-cards")

        msg = str(exc_info.value)
        assert "500" in msg
        # No raw control characters in the surfaced message.
        assert "\x1b" not in msg
        assert "\x00" not in msg
        assert "\r" not in msg
        # Bounded length — never the full 4 KB body.
        assert len(msg) < 1024
