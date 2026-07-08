"""RemoteMemoryStore — HTTP skeleton for a shared card service.

Card CRUD speaks Card JSON (``model_dump`` out, ``model_validate`` in) to a
remote service. Retrieval has no remote counterpart yet: ``nearest`` and
``research`` raise until that port lands.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.storage.base import (
    MemoryStore,
    ResearchRequest,
    ResearchResult,
    ScoredCard,
)


class RemoteMemoryStore(MemoryStore):
    def __init__(
        self,
        base_url: str = "",
        timeout_s: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        if client is None and not base_url:
            raise ValueError("either base_url or an httpx client is required")
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    @property
    def is_ready(self) -> bool:
        try:
            return self._client.get("/health").is_success
        except httpx.HTTPError:
            return False

    def save(self, card: Card) -> str:
        response = self._client.post("/cards", json=card.model_dump(mode="json"))
        response.raise_for_status()
        return str(response.json()["id"])

    def get(self, card_id: str) -> Card | None:
        response = self._client.get(f"/cards/{card_id}")
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        response.raise_for_status()
        return Card.model_validate(response.json())

    def delete(self, card_id: str) -> bool:
        response = self._client.delete(f"/cards/{card_id}")
        if response.status_code == httpx.codes.NOT_FOUND:
            return False
        response.raise_for_status()
        return True

    def snapshot(self) -> tuple[Card, ...]:
        response = self._client.get("/cards")
        response.raise_for_status()
        cards = (Card.model_validate(data) for data in response.json()["cards"])
        return tuple(sorted(cards, key=lambda card: card.id))

    def apply_merges(self, merged: Sequence[Card]) -> list[str]:
        if not merged:
            return []
        response = self._client.post(
            "/cards/merge",
            json={"cards": [card.model_dump(mode="json") for card in merged]},
        )
        response.raise_for_status()
        return [str(card_id) for card_id in response.json()["ids"]]

    def nearest(
        self, text: str, k: int, kind: CardKind | None = None
    ) -> list[ScoredCard]:
        raise NotImplementedError("remote retrieval has not landed yet")

    async def research(self, request: ResearchRequest) -> ResearchResult:
        raise NotImplementedError("remote retrieval has not landed yet")

    def rebuild(self) -> None:
        self._client.post("/rebuild").raise_for_status()

    def close(self) -> None:
        self._client.close()
