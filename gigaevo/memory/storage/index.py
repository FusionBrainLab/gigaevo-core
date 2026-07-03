"""Chroma-backed vector index over configured embed scopes.

The only module in the memory system that touches Chroma or embeddings.
One persistent client, one collection per scope; each collection holds one
document per card — the labeled concatenation of that scope's card fields.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import threading
from typing import Any, cast

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pydantic import BaseModel, ConfigDict

from gigaevo.exceptions import StorageError
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.storage.config import EmbedConfig
from gigaevo.memory.storage.hf_cache import ensure_writable_hf_cache

# Written beside the Chroma data. Records the embedding config the persisted
# vectors were built with, so reopening the dir under a changed embedder fails
# loudly instead of ranking new queries against incompatible stored vectors.
_FINGERPRINT_FILE = "embed_fingerprint.json"


def _embed_fingerprint(embed: EmbedConfig) -> dict[str, Any]:
    """The embedding settings that determine the stored vectors: the model and
    each scope's field set. query_prefix/nearest_scope condition only queries,
    never the indexed documents, so they are deliberately excluded."""
    return {
        "embedding_model": embed.embedding_model,
        "embed_scopes": {
            scope: list(fields) for scope, fields in sorted(embed.embed_scopes.items())
        },
    }


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
        # The reader queries on the event loop while a live restamp upserts on a
        # to_thread worker — serialize all Chroma access so concurrent
        # query/upsert/remove on the one client cannot race.
        self._lock = threading.Lock()
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._guard_embed_fingerprint(Path(persist_dir), embed)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        # sentence-transformers follows HF_HOME and friends; redirect them to a
        # writable dir before the model download begins.
        ensure_writable_hf_cache()
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embed.embedding_model
        )
        self._collections = {
            scope: self._client.get_or_create_collection(
                name=f"cards_{scope}", embedding_function=cast(Any, embedding_fn)
            )
            for scope in embed.embed_scopes
        }

    @staticmethod
    def _guard_embed_fingerprint(persist_dir: Path, embed: EmbedConfig) -> None:
        """Refuse to reopen a persist dir whose vectors were built with a
        different embedding config; stamp the fingerprint on first use.

        Chroma keys collections by name only, so a reused dir keeps the old
        embedder's vectors even after ``rebuild`` (which diffs by document
        text). Ranking new-embedder queries against them silently corrupts
        retrieval — or hard-fails on a dimension mismatch. Reject up front and
        point the user at a fresh ``checkpoint_dir``.
        """
        fingerprint = _embed_fingerprint(embed)
        path = persist_dir / _FINGERPRINT_FILE
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                # A present-but-unreadable fingerprint is more suspicious than an
                # absent one (a truncated write, a foreign file) — fail closed
                # rather than silently re-stamp and rank against unknown vectors.
                raise StorageError(
                    f"Unreadable embed fingerprint at {path}: {exc}. Cannot verify "
                    f"the persisted vectors match the run's embedder — use a fresh "
                    f"checkpoint_dir."
                ) from exc
            if existing != fingerprint:
                raise StorageError(
                    f"Embedding config changed for memory index {persist_dir}: "
                    f"persisted vectors were built with {existing}, but the run "
                    f"requests {fingerprint}. The old vectors are incompatible "
                    f"with the new embedder — use a fresh checkpoint_dir."
                )
        # Atomic stamp: a crashed write leaves the prior fingerprint intact
        # instead of a truncated file the guard would now reject.
        tmp = persist_dir / f"{_FINGERPRINT_FILE}.{os.getpid()}.tmp"
        try:
            tmp.write_text(json.dumps(fingerprint, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

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
        with self._lock:
            if k <= 0 or not text.strip() or collection.count() == 0:
                return []
            # Asymmetric embedding: the retrieval query carries the embedder's
            # query instruction; the indexed card documents never do (upsert
            # embeds render_scope_document verbatim). Empty prefix is a no-op.
            result = collection.query(
                query_texts=[f"{self._embed.query_prefix}{text}"],
                n_results=k,
                where=self._where(kind, exclude_ids),
                include=["distances"],
            )
        ids = result["ids"][0]
        result_distances = result["distances"]
        distances = result_distances[0] if result_distances else []
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
