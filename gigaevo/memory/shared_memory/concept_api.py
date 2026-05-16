"""HTTP client for the Memory API entity endpoints.

Sync client backed by ``requests.Session`` over ``urllib3``'s connection
pool. Retries on transient 5xx responses are configured inside
``urllib3`` itself via :func:`gigaevo.infra.requests_factory.build_retry`
so they happen before responses reach this module.

Despite the class name, the client targets the newer Memory API entity
model:
- memory cards: ``/v1/memory-cards``
- search: ``/v1/search/batch``
"""

from __future__ import annotations

import re
from typing import Any

import requests

from gigaevo.exceptions import MemoryStorageError
from gigaevo.infra.requests_factory import make_requests_session

# Response bodies from a misbehaving server can be arbitrarily large or
# carry control characters; either property is unsafe to splice raw into
# an exception message that ends up in log sinks.  The preview helper
# strips ANSI/CSI escape sequences as whole units, drops residual C0/C1
# control bytes, then truncates.
_RESPONSE_BODY_PREVIEW_BYTES = 512

# Strip ANSI/CSI escape sequences as a unit so the trailing parameter
# bytes (``[31m``, ``[0;1;33;45m`` …) don't survive as printable text.
# A per-char filter only drops the ESC (``\x1b``) byte and leaves the
# whole readable-but-meaningless CSI tail in place.
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _safe_response_preview(response: requests.Response) -> str:
    """Return a short, control-stripped preview of a response body.

    Used in exception messages so transient server failures stay legible
    without leaking ANSI / CRLF / RLO sequences into log sinks (a sink
    rendering raw bytes from an attacker-controlled body is the same
    class of vector this codebase has hardened elsewhere — see T-C1).

    Strip controls *before* truncating: if the first 512 bytes of a 4 KB
    response are an ANSI escape burst with the readable error message
    trailing, truncating raw bytes first would drop the message before
    stripping has a chance to compress the leading escapes out."""
    try:
        text = response.text
    except Exception:
        return "<unreadable body>"
    # 1. Drop ANSI/CSI escape sequences as a unit.
    text = _ANSI_CSI_RE.sub("", text)
    # 2. Permit printable ASCII + ``\n`` + non-control Unicode (``>= U+00A0``
    #    keeps NBSP and everything beyond, including legitimate non-Latin
    #    text).  Strip C0 (incl. DEL=0x7F) and C1 (0x80–0x9F) controls.
    sanitised = "".join(
        ch for ch in text if ch == "\n" or 0x20 <= ord(ch) < 0x7F or ord(ch) >= 0xA0
    )
    # 3. Truncate the now-sanitised string.
    if len(sanitised) > _RESPONSE_BODY_PREVIEW_BYTES:
        sanitised = sanitised[:_RESPONSE_BODY_PREVIEW_BYTES] + "… (truncated)"
    return sanitised


class _ConceptApiClient:
    """Small HTTP client around Memory API entity endpoints."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._http = make_requests_session("memory_api", timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any] | None:
        url = f"{self._base_url}{path}"
        try:
            response = self._http.request(method, url, **kwargs)
        except requests.exceptions.ConnectionError as exc:
            raise MemoryStorageError(
                f"Cannot connect to Memory API at {self._base_url}: {exc}. "
                "Start the API service or set MEMORY_API_URL to a reachable endpoint."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise MemoryStorageError(
                f"Memory API request timed out for {self._base_url}: {exc}. "
                "Check service health and network connectivity."
            ) from exc
        if response.status_code == 204:
            return None
        if response.status_code >= 400:
            raise MemoryStorageError(
                f"Memory API request failed ({method} {path}): "
                f"{response.status_code} {_safe_response_preview(response)}"
            )
        return response.json()

    def save_concept(
        self,
        *,
        content: dict[str, Any],
        name: str,
        tags: list[str],
        when_to_use: str,
        channel: str,
        namespace: str | None,
        author: str | None,
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        """Create or update a memory card via the API.

        Uses POST for new cards, PUT for existing (when entity_id is given).

        Returns:
            API response with ``entity_id`` and ``version_id``.
        """
        body = {
            "meta": {
                "name": name,
                "tags": tags,
                "when_to_use": when_to_use,
                "namespace": namespace,
                "author": author,
            },
            "channel": channel,
            "content": content,
        }
        if entity_id:
            result = self._request("PUT", f"/v1/memory-cards/{entity_id}", json=body)
        else:
            result = self._request("POST", "/v1/memory-cards", json=body)
        if not isinstance(result, dict):
            raise MemoryStorageError("Unexpected empty response from concept save")
        return result

    def get_concept(self, entity_id: str, channel: str = "latest") -> dict[str, Any]:
        """Fetch a single memory card by entity ID."""
        result = self._request(
            "GET",
            f"/v1/memory-cards/{entity_id}",
            params={"channel": channel},
        )
        if not isinstance(result, dict):
            raise MemoryStorageError("Unexpected empty response from concept get")
        return result

    def list_memory_cards(
        self,
        *,
        limit: int,
        offset: int = 0,
        channel: str = "latest",
    ) -> list[dict[str, Any]]:
        """Paginated listing of all memory cards. Returns dicts, filters non-dicts."""
        result = self._request(
            "GET",
            "/v1/memory-cards",
            params={"limit": int(limit), "offset": int(offset), "channel": channel},
        )
        if not isinstance(result, list):
            return []
        items: list[dict[str, Any]] = []
        for row in result:
            if isinstance(row, dict):
                items.append(row)
        return items

    def search_concepts(
        self,
        *,
        query: str | None,
        limit: int,
        namespace: str | None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Batch semantic search. Returns ``{"hits": [...], "total": N}``."""
        query_text = str(query or "").strip()
        if not query_text:
            return {"hits": [], "total": 0}

        payload: dict[str, Any] = {
            "queries": [query_text],
            "top_k": int(limit),
            "entity_type": "memory_card",
            "channel": "latest",
            "search_type": "bm25",
        }
        if namespace:
            payload["namespace"] = namespace

        result = self._request("POST", "/v1/search/batch", json=payload) or {}
        if not isinstance(result, dict):
            return {"hits": [], "total": 0}

        results = result.get("results")
        if not isinstance(results, list) or not results:
            return {"hits": [], "total": 0}
        first = results[0]
        if not isinstance(first, list):
            return {"hits": [], "total": 0}

        hits: list[dict[str, Any]] = [h for h in first if isinstance(h, dict)]
        return {"hits": hits, "total": len(hits)}

    def delete_concept(self, entity_id: str) -> None:
        """Delete a memory card by entity ID."""
        self._request("DELETE", f"/v1/memory-cards/{entity_id}")
