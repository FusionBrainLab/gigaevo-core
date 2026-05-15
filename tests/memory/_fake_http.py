"""Test shim for :class:`_ConceptApiClient` after the ``httpx`` → ``requests`` swap.

Mirrors the surface tests previously relied on with ``httpx.MockTransport`` —
handler callbacks receive a request-like object and return a response-like
object — but uses ``unittest.mock`` to patch ``requests.Session.request``
on a private ``_ConceptApiClient`` instance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json as _json
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import parse_qsl, urlsplit

import requests

from gigaevo.memory.shared_memory.concept_api import _ConceptApiClient


class _FakeUrl(str):
    """``str`` subclass that also exposes ``.params`` like ``httpx.URL``."""

    params: dict[str, str]

    def __new__(cls, value: str, params: dict[str, str] | None = None) -> _FakeUrl:
        instance = super().__new__(cls, value)
        instance.params = params or {}
        return instance


@dataclass
class _FakeRequest:
    """Minimal request stand-in matching the attrs tests previously asserted on."""

    method: str
    url: _FakeUrl
    content: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


def make_fake_response(
    status_code: int = 200,
    *,
    json: Any = None,
    text: str = "",
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """Build a ``requests.Response`` mirroring ``httpx.Response`` constructor signature."""
    response = requests.Response()
    response.status_code = status_code
    if json is not None:
        response._content = _json.dumps(json).encode("utf-8")
        response.headers["Content-Type"] = "application/json"
    else:
        response._content = text.encode("utf-8") if text else b""
    if headers:
        response.headers.update(headers)
    return response


def make_mocked_client(
    handler: Callable[[_FakeRequest], requests.Response],
    *,
    base_url: str = "http://test:8000",
) -> _ConceptApiClient:
    """Build a ``_ConceptApiClient`` whose ``_http.request`` is routed through ``handler``.

    Handler receives :class:`_FakeRequest` (with ``method``, ``url``, ``content``
    populated from the captured kwargs) and returns a ``requests.Response`` —
    typically built via :func:`make_fake_response`.
    """
    client = _ConceptApiClient.__new__(_ConceptApiClient)
    client._base_url = base_url.rstrip("/")

    fake_session = MagicMock(spec=requests.Session)

    def _dispatch(method: str, url: str, **kwargs: Any) -> requests.Response:
        body_json = kwargs.get("json")
        body_data = kwargs.get("data")
        content: bytes
        if body_json is not None:
            content = _json.dumps(body_json).encode("utf-8")
        elif isinstance(body_data, (bytes, bytearray)):
            content = bytes(body_data)
        elif body_data is not None:
            content = str(body_data).encode("utf-8")
        else:
            content = b""

        parts = urlsplit(url)
        params_from_url = dict(parse_qsl(parts.query))
        merged_params = {**params_from_url, **(kwargs.get("params") or {})}
        merged_params = {str(k): str(v) for k, v in merged_params.items()}
        fake_url = _FakeUrl(url, params=merged_params)

        request = _FakeRequest(
            method=method,
            url=fake_url,
            content=content,
            headers=dict(kwargs.get("headers") or {}),
        )
        return handler(request)

    fake_session.request = MagicMock(side_effect=_dispatch)
    client._http = fake_session
    return client
