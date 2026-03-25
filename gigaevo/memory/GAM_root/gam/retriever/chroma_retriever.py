# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
from typing import Any, Dict, List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from GAM_root.gam.retriever.base import AbsRetriever
from GAM_root.gam.schemas import Hit


class ChromaRetriever(AbsRetriever):
    name = "vector"
    _DEFAULT_FIELD_COLLECTIONS = {
        "description": "memories_description",
        "task_description": "memories_task_description",
        "explanation_summary": "memories_explanation_summary",
        "description_explanation_summary": "memories_description_explanation_summary",
        "description_task_description_summary": "memories_description_task_description_summary",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        persist_dir = config.get("persist_dir")
        if not persist_dir:
            raise ValueError("ChromaRetriever requires 'persist_dir' in config")

        self.model_name = config.get("model_name", "all-MiniLM-L6-v2")
        self.source_label = str(config.get("source_label") or self.name)
        configured_names = config.get("collection_names")
        if not isinstance(configured_names, dict):
            configured_names = {}
        self.collection_names = {
            field: str(configured_names.get(field) or default_name)
            for field, default_name in self._DEFAULT_FIELD_COLLECTIONS.items()
        }

        requested_collections = config.get("active_collections")
        if isinstance(requested_collections, list) and requested_collections:
            active = []
            for field in requested_collections:
                name = str(field).strip()
                if name in self.collection_names and name not in active:
                    active.append(name)
            self.active_collections = active or list(self.collection_names.keys())
        else:
            self.active_collections = list(self.collection_names.keys())

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )
        self.collections = {
            field: self.client.get_or_create_collection(
                name=self.collection_names[field],
                embedding_function=self.embedding_function,
            )
            for field in self.active_collections
        }

    def build(self, page_store) -> None:
        pages = page_store.load() if page_store is not None else []
        payload_by_field: Dict[str, Dict[str, Dict[str, Any]]] = {
            field: {}
            for field in self.active_collections
        }

        for idx, page in enumerate(pages):
            card = self._extract_card(page)
            doc_id = self._resolve_doc_id(page, card, idx)
            if not doc_id:
                continue

            texts = self._extract_field_texts(page, card)
            compact_snippet = self._build_compact_snippet(
                description=str(texts.get("description") or ""),
                explanation_summary=str(texts.get("explanation_summary") or ""),
            )
            for field in self.active_collections:
                document = str(texts.get(field) or "").strip()
                if not document:
                    continue
                payload_by_field[field][doc_id] = {
                    "document": document,
                    "metadata": {
                        "amem_id": doc_id,
                        "vector_field": field,
                        "content": compact_snippet,
                        "header": str(getattr(page, "header", "") or ""),
                    },
                }

        for field, collection in self.collections.items():
            self._sync_collection(collection, payload_by_field.get(field, {}))

    def load(self) -> None:
        return None

    def update(self, page_store) -> None:
        self.build(page_store)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        if not query_list:
            return []

        hits_per_query: List[List[Hit]] = []
        for query in query_list:
            aggregated: Dict[str, Hit] = {}

            for field, collection in self.collections.items():
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                ids = results.get("ids", [[]])
                docs = results.get("documents", [[]])
                metas = results.get("metadatas", [[]])
                dists = results.get("distances", [[]])

                q_ids = ids[0] if ids else []
                q_docs = docs[0] if docs else []
                q_metas = metas[0] if metas else []
                q_dists = dists[0] if dists else []

                for i, raw_doc_id in enumerate(q_ids):
                    doc_id = str(raw_doc_id or "").strip()
                    if not doc_id:
                        continue

                    metadata: Dict[str, Any] = {}
                    if i < len(q_metas) and isinstance(q_metas[i], dict):
                        metadata = self._convert_metadata_types(q_metas[i])

                    document = q_docs[i] if i < len(q_docs) else ""
                    distance = q_dists[i] if i < len(q_dists) else None
                    score = self._distance_to_score(distance)
                    snippet = str(metadata.get("content") or document or "")

                    hit = Hit(
                        page_id=doc_id,
                        snippet=snippet,
                        source=self.source_label,
                        meta={
                            "score": score,
                            "distance": distance,
                            "field": field,
                            "collection": self.collection_names.get(field, ""),
                            "metadata": metadata,
                        },
                    )
                    if document:
                        hit.meta["document"] = document

                    existing = aggregated.get(doc_id)
                    existing_score = existing.meta.get("score", 0.0) if existing else -1.0
                    if existing is None or score > existing_score:
                        aggregated[doc_id] = hit

            merged_hits = sorted(
                aggregated.values(),
                key=lambda h: float(h.meta.get("score", 0.0)),
                reverse=True,
            )[:top_k]
            hits_per_query.append(merged_hits)

        return hits_per_query

    def _sync_collection(self, collection: Any, payload: Dict[str, Dict[str, Any]]) -> None:
        current_ids = collection.get().get("ids", []) or []
        current_ids_set = {str(doc_id) for doc_id in current_ids if str(doc_id).strip()}
        next_ids_set = set(payload.keys())

        stale_ids = list(current_ids_set - next_ids_set)
        if stale_ids:
            collection.delete(ids=stale_ids)

        if not payload:
            return

        ids = list(payload.keys())
        documents = [payload[doc_id]["document"] for doc_id in ids]
        metadatas = [payload[doc_id]["metadata"] for doc_id in ids]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    @staticmethod
    def _distance_to_score(distance: Any) -> float:
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0.0
        if value < 0:
            return 0.0
        return 1.0 / (1.0 + value)

    @staticmethod
    def _extract_card(page: Any) -> Dict[str, Any]:
        meta = getattr(page, "meta", None)
        if not isinstance(meta, dict):
            return {}
        record = meta.get("amem")
        if isinstance(record, dict):
            return record
        return {}

    @staticmethod
    def _resolve_doc_id(page: Any, card: Dict[str, Any], fallback_idx: int) -> str:
        page_meta = getattr(page, "meta", None)
        if isinstance(page_meta, dict):
            amem_id = str(page_meta.get("amem_id") or "").strip()
            if amem_id:
                return amem_id
        card_id = str(card.get("id") or "").strip()
        if card_id:
            return card_id
        return str(fallback_idx)

    def _extract_field_texts(self, page: Any, card: Dict[str, Any]) -> Dict[str, str]:
        parsed = self._parse_labeled_content(str(getattr(page, "content", "") or ""))
        explanation = card.get("explanation")
        if isinstance(explanation, dict):
            explanation_summary = str(explanation.get("summary") or "")
        else:
            explanation_summary = str(explanation or "")

        description = str(card.get("description") or card.get("content") or parsed.get("description") or "")
        task_description = str(
            card.get("task_description_summary")
            or parsed.get("task_description_summary")
            or card.get("task_description")
            or card.get("context")
            or parsed.get("task_description")
            or ""
        )
        explanation_summary = str(
            explanation_summary
            or card.get("explanation_summary")
            or parsed.get("explanation_summary")
            or ""
        )
        description_explanation_summary = self._merge_labeled_text(
            description,
            explanation_summary,
            first_label="IDEA_DESCRIPTION",
            second_label="EXPLANATION_SUMMARY",
        )
        description_task_description_summary = self._merge_labeled_text(
            description,
            task_description,
            first_label="IDEA_DESCRIPTION",
            second_label="TASK_DESCRIPTION_SUMMARY",
        )
        return {
            "description": description,
            "task_description": task_description,
            "explanation_summary": explanation_summary,
            "description_explanation_summary": description_explanation_summary,
            "description_task_description_summary": description_task_description_summary,
        }

    @staticmethod
    def _build_compact_snippet(description: str, explanation_summary: str) -> str:
        description_text = " ".join(str(description or "").split())
        explanation_text = " ".join(str(explanation_summary or "").split())

        if description_text and explanation_text:
            return (
                f"IDEA_DESCRIPTION: {description_text}\n"
                f"EXPLANATION_SUMMARY: {explanation_text}"
            )
        if description_text:
            return f"IDEA_DESCRIPTION: {description_text}"
        if explanation_text:
            return f"EXPLANATION_SUMMARY: {explanation_text}"
        return ""

    @staticmethod
    def _merge_labeled_text(
        first: str,
        second: str,
        *,
        first_label: str,
        second_label: str,
    ) -> str:
        first_text = " ".join(str(first or "").split())
        second_text = " ".join(str(second or "").split())

        if first_text and second_text:
            return f"{first_label}: {first_text}\n{second_label}: {second_text}"
        if first_text:
            return f"{first_label}: {first_text}"
        if second_text:
            return f"{second_label}: {second_text}"
        return ""

    @staticmethod
    def _parse_labeled_content(content: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if not key:
                continue
            out[key] = value.strip()
        return out

    def _convert_metadata_types(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        converted = dict(metadata)
        for key, value in converted.items():
            if not isinstance(value, str):
                continue
            if key in {"content", "header", "vector_field", "amem_id"}:
                continue
            try:
                converted[key] = ast.literal_eval(value)
            except Exception:
                pass
        return converted
