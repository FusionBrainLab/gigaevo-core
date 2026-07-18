"""In-memory Chroma vector index over configured embed scopes.

The only module in the memory system that touches Chroma or embeddings.
Each process derives one collection per scope from its authoritative card bank;
each collection holds one document per card — the labeled concatenation of that
scope's card fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import threading
from typing import Any, cast
from uuid import uuid4

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np
from pydantic import BaseModel, ConfigDict

from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.storage.config import EmbedConfig
from gigaevo.memory.storage.hf_cache import ensure_writable_hf_cache


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
    def __init__(self, embed: EmbedConfig) -> None:
        self._embed = embed
        # The reader queries on the event loop while a live restamp upserts on a
        # to_thread worker — serialize all Chroma access so concurrent
        # query/upsert/remove on the one client cannot race.
        self._lock = threading.Lock()
        self._client = chromadb.EphemeralClient()
        # sentence-transformers follows HF_HOME and friends; redirect them to a
        # writable dir before the model download begins.
        ensure_writable_hf_cache()
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embed.embedding_model
        )
        namespace = uuid4().hex
        self._collections = {
            scope: self._client.get_or_create_collection(
                name=f"cards_{scope}_{namespace}",
                embedding_function=cast(Any, self._embedding_fn),
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
                    {
                        "card_id": card.id,
                        "kind": card.kind.value,
                        "task_key": card.task_key,
                    },
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
        task_key: str | None = None,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> list[IndexHit]:
        """The ``k`` closest cards to ``text`` in a scope, ascending distance."""
        if scope not in self._collections:
            raise KeyError(
                f"unknown embed scope {scope!r}; configured: {sorted(self._collections)}"
            )
        collection = self._collections[scope]
        with self._lock:
            if k <= 0 or not text.strip() or collection.count() == 0:
                return []
            # Asymmetric embedding: the retrieval query carries the embedder's
            # query instruction; the indexed card documents never do (upsert
            # embeds render_scope_document verbatim). Empty prefix is a no-op.
            result = collection.query(
                query_texts=[f"{self._embed.query_prefix}{text}"],
                n_results=k,
                where=self._where(kind, task_key, exclude_ids),
                include=["distances"],
            )
        ids = result["ids"][0]
        result_distances = result["distances"]
        distances = result_distances[0] if result_distances else []
        return [
            IndexHit(card_id=cid, distance=float(dist))
            for cid, dist in zip(ids, distances, strict=True)
        ]

    def mmr_order(
        self,
        scope: str,
        text: str,
        card_ids: Sequence[str],
        *,
        lambda_: float = 1.0,
        relevance: Mapping[str, float] | None = None,
    ) -> list[str]:
        """Greedy maximal-marginal-relevance ordering of ``card_ids``.

        Each step picks ``argmax lambda_ * rel - (1 - lambda_) * max_sim`` over
        the not-yet-picked cards, where ``rel`` is cosine similarity to ``text``
        (or the caller-supplied ``relevance`` score) and ``max_sim`` is the
        cosine similarity to the closest already-picked card. ``lambda_=1.0``
        is pure relevance order; lower values penalize near-duplicates. Ids
        without a stored embedding keep their input order at the tail; ties
        resolve to input order.
        """
        if scope not in self._collections:
            raise KeyError(
                f"unknown embed scope {scope!r}; configured: {sorted(self._collections)}"
            )
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0.0, 1.0], got {lambda_}")
        unique_ids = list(dict.fromkeys(card_ids))
        if not unique_ids:
            return []
        collection = self._collections[scope]
        with self._lock:
            stored = collection.get(ids=unique_ids, include=["embeddings"])
            query_embedding = (
                None
                if relevance is not None
                else self._embedding_fn([f"{self._embed.query_prefix}{text}"])[0]
            )
        embeddings = stored["embeddings"]
        vectors = {
            cid: np.asarray(vector, dtype=float)
            for cid, vector in zip(
                stored["ids"],
                embeddings if embeddings is not None else [],
                strict=True,
            )
        }
        known = [cid for cid in unique_ids if cid in vectors]
        missing = [cid for cid in unique_ids if cid not in vectors]
        if len(known) <= 1:
            return known + missing
        matrix = np.stack([vectors[cid] for cid in known])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.where(norms == 0.0, 1.0, norms)
        if relevance is not None:
            rel = np.asarray([float(relevance.get(cid, 0.0)) for cid in known])
        else:
            query = np.asarray(query_embedding, dtype=float)
            query = query / (float(np.linalg.norm(query)) or 1.0)
            rel = matrix @ query
        pairwise = matrix @ matrix.T
        picked: list[int] = []
        remaining = list(range(len(known)))
        while remaining:
            if picked:
                max_sim = pairwise[np.ix_(remaining, picked)].max(axis=1)
            else:
                max_sim = np.zeros(len(remaining))
            scores = lambda_ * rel[remaining] - (1.0 - lambda_) * max_sim
            picked.append(remaining.pop(int(np.argmax(scores))))
        return [known[i] for i in picked] + missing

    @staticmethod
    def _where(
        kind: CardKind | None,
        task_key: str | None,
        exclude_ids: frozenset[str],
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if kind is not None:
            clauses.append({"kind": kind.value})
        if task_key is not None:
            clauses.append({"task_key": task_key})
        if exclude_ids:
            clauses.append({"card_id": {"$nin": sorted(exclude_ids)}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
