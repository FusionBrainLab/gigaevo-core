"""Chroma-backed vector index over configured embed scopes.

The only module in the memory system that touches Chroma or embeddings.
One persistent client, one collection per scope; each collection holds one
document per card — the labeled concatenation of that scope's card fields.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pydantic import BaseModel, ConfigDict

from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.storage.config import EmbedConfig


class IndexHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    card_id: str
    distance: float


def render_scope_document(card: Card, fields: Sequence[str]) -> str:
    """The text embedded for a card under a scope.

    A single-field scope embeds the raw field; a multi-field scope embeds
    ``FIELD_NAME: value`` lines so the embedding keeps field identity.
    Empty fields are skipped; an all-empty card yields "" (not indexed).
    """
    values = [(f, str(getattr(card, f)).strip()) for f in fields]
    values = [(f, v) for f, v in values if v]
    if not values:
        return ""
    if len(fields) == 1:
        return values[0][1]
    return "\n".join(f"{f.upper()}: {v}" for f, v in values)


class VectorIndex:
    def __init__(self, persist_dir: str | Path, embed: EmbedConfig) -> None:
        self._embed = embed
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embed.embedding_model
        )
        self._collections = {
            scope: self._client.get_or_create_collection(
                name=f"cards_{scope}", embedding_function=embedding_fn
            )
            for scope in embed.embed_scopes
        }

    @property
    def scopes(self) -> tuple[str, ...]:
        return tuple(self._collections)

    def rebuild(self, cards: Sequence[Card]) -> None:
        """Make every collection reflect exactly ``cards``.

        Diff-sync: stale documents are deleted, and only missing or changed
        documents are re-embedded — a rebuild over an unchanged bank is cheap.
        """
        for scope, collection in self._collections.items():
            desired = self._desired_documents(scope, cards)
            existing = collection.get(include=["documents"])
            existing_docs = dict(
                zip(existing["ids"], existing["documents"] or [], strict=True)
            )
            stale = sorted(set(existing_docs) - desired.keys())
            if stale:
                collection.delete(ids=stale)
            changed = sorted(
                cid
                for cid, (document, _) in desired.items()
                if existing_docs.get(cid) != document
            )
            if changed:
                collection.upsert(
                    ids=changed,
                    documents=[desired[i][0] for i in changed],
                    metadatas=[desired[i][1] for i in changed],
                )

    def upsert(self, cards: Sequence[Card]) -> None:
        """Index these cards in every scope (re-embedding them); a card whose
        scope document is empty is dropped from that scope."""
        if not cards:
            return
        card_ids = [card.id for card in cards]
        for scope, collection in self._collections.items():
            desired = self._desired_documents(scope, cards)
            emptied = [cid for cid in card_ids if cid not in desired]
            self._delete_present(collection, emptied)
            if desired:
                ids = sorted(desired)
                collection.upsert(
                    ids=ids,
                    documents=[desired[i][0] for i in ids],
                    metadatas=[desired[i][1] for i in ids],
                )

    def remove(self, card_ids: Sequence[str]) -> None:
        for collection in self._collections.values():
            self._delete_present(collection, list(card_ids))

    def _desired_documents(
        self, scope: str, cards: Sequence[Card]
    ) -> dict[str, tuple[str, dict[str, str]]]:
        fields = self._embed.embed_scopes[scope]
        desired: dict[str, tuple[str, dict[str, str]]] = {}
        for card in cards:
            document = render_scope_document(card, fields)
            if document:
                desired[card.id] = (
                    document,
                    {"card_id": card.id, "kind": card.kind.value},
                )
        return desired

    @staticmethod
    def _delete_present(collection: Any, card_ids: list[str]) -> None:
        if not card_ids:
            return
        present = collection.get(ids=card_ids, include=[])["ids"]
        if present:
            collection.delete(ids=list(present))

    def query(
        self,
        scope: str,
        text: str,
        k: int,
        *,
        kind: CardKind | None = None,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> list[IndexHit]:
        """The ``k`` closest cards to ``text`` in a scope, ascending distance."""
        if scope not in self._collections:
            raise KeyError(
                f"unknown embed scope {scope!r}; configured: {sorted(self._collections)}"
            )
        collection = self._collections[scope]
        if k <= 0 or not text.strip() or collection.count() == 0:
            return []
        result = collection.query(
            query_texts=[text],
            n_results=k,
            where=self._where(kind, exclude_ids),
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [
            IndexHit(card_id=cid, distance=float(dist))
            for cid, dist in zip(ids, distances, strict=True)
        ]

    @staticmethod
    def _where(
        kind: CardKind | None, exclude_ids: frozenset[str]
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if kind is not None:
            clauses.append({"kind": kind.value})
        if exclude_ids:
            clauses.append({"card_id": {"$nin": sorted(exclude_ids)}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
