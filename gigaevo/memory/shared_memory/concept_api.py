"""HTTP client for the Memory API entity endpoints.

Extracted from memory.py for cleaner modularity.
"""

from __future__ import annotations

from typing import Any

import httpx

from gigaevo.exceptions import MemoryStorageError


class _ConceptApiClient:
    """Small HTTP client around Memory API entity endpoints.

    NOTE: despite the class name, this client now targets the newer Memory API
    entity model:
    - memory cards: ``/v1/memory-cards``
    - search: ``/v1/search/batch``
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        base = base_url.rstrip("/")
        self._http = httpx.Client(base_url=base, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any] | None:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            host = str(self._http.base_url).rstrip("/")
            raise MemoryStorageError(
                f"Cannot connect to Memory API at {host}. "
                "Start the API service or set MEMORY_API_URL to a reachable endpoint."
            ) from exc
        except httpx.TimeoutException as exc:
            host = str(self._http.base_url).rstrip("/")
            raise MemoryStorageError(
                f"Memory API request timed out for {host}. "
                "Check service health and network connectivity."
            ) from exc
        if response.status_code == 204:
            return None
        if response.status_code >= 400:
            raise MemoryStorageError(
                f"Memory API request failed ({method} {path}): "
                f"{response.status_code} {response.text}"
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
        self._request("DELETE", f"/v1/memory-cards/{entity_id}")
