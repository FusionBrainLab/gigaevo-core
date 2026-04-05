"""Card deduplication pipeline: vector scoring, LLM decision, merge computation."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

from loguru import logger

from gigaevo.memory.shared_memory.card_conversion import (
    AnyCard,
    is_program_card,
    normalize_memory_card,
)
from gigaevo.memory.shared_memory.card_store import CardStore
from gigaevo.memory.shared_memory.card_update_dedup import (
    QUERY_DESCRIPTION,
    QUERY_DESCRIPTION_EXPLANATION_SUMMARY,
    QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY,
    QUERY_EXPLANATION_SUMMARY,
    CardUpdateDedupConfig,
    build_dedup_queries,
    compute_weighted_candidates,
    get_explanation_summary,
    get_full_explanations,
    merge_updated_card,
    parse_llm_card_decision,
)
from gigaevo.memory.shared_memory.utils import truncate_text


class CardDedup:
    """Card deduplication: vector scoring, LLM decision, merge computation.

    Does NOT write cards — returns merge results for the orchestrator.
    """

    def __init__(
        self,
        card_store: CardStore,
        llm_service: Any,
        config: CardUpdateDedupConfig,
        allowed_gam_tools: set[str],
        gam_store_dir: Path,
        export_file: Path,
        checkpoint_dir: Path,
    ):
        self._card_store = card_store
        self.llm_service = llm_service
        self._config = config
        self._allowed_gam_tools = allowed_gam_tools
        self._gam_store_dir = gam_store_dir
        self._export_file = export_file
        self._checkpoint_dir = checkpoint_dir
        self._retrievers: dict[str, Any] | None = None

    @property
    def config(self) -> CardUpdateDedupConfig:
        return self._config

    def invalidate_retrievers(self) -> None:
        self._retrievers = None

    # --- Retriever building ---

    def build_retrievers(self) -> dict[str, Any]:
        """Build dedup retriever index from exported records."""
        try:
            from gigaevo.memory.shared_memory.amem_gam_retriever import (
                build_gam_store,
                build_retrievers,
                load_amem_records,
            )
        except Exception as exc:
            logger.warning("[Memory] Dedup retriever import failed: {}", exc)
            return {}

        self._gam_store_dir.mkdir(parents=True, exist_ok=True)
        if self._export_file.exists():
            try:
                records = load_amem_records(self._export_file)
            except Exception:
                records = [c.model_dump() for c in self._card_store.cards.values()]
        else:
            records = [c.model_dump() for c in self._card_store.cards.values()]
        records = [
            r
            for r in records
            if str(r.get("category", "")).strip().lower() != "program"
        ]
        if not records:
            return {}

        try:
            _, page_store, _ = build_gam_store(records, self._gam_store_dir)
            retrievers = build_retrievers(
                page_store,
                self._gam_store_dir / "indexes",
                self._checkpoint_dir / "chroma",
                enable_bm25=False,
                allowed_tools=[
                    "vector_description",
                    "vector_explanation_summary",
                    "vector_description_explanation_summary",
                    "vector_description_task_description_summary",
                ],
            )
        except Exception as exc:
            logger.warning("[Memory] Dedup retriever build failed: {}", exc)
            return {}

        return {
            name: retriever
            for name, retriever in retrievers.items()
            if name in self._allowed_gam_tools
        }

    def resolve_retriever(self, tool_name: str) -> Any:
        """Resolve a retriever by tool name, building lazily if needed."""
        if self._retrievers is None:
            self._retrievers = self.build_retrievers()
        retrievers = self._retrievers or {}
        if not retrievers:
            return None

        retriever = retrievers.get(tool_name)
        if retriever is None and tool_name != "vector":
            retriever = retrievers.get("vector")
        return retriever

    # --- Scoring ---

    def score_candidates(
        self,
        card: AnyCard,
        resolve_retriever_fn: Callable[[str], Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Score existing cards against incoming card via vector search.

        ``resolve_retriever_fn`` resolves tool names to retrievers.
        If not provided, uses ``self.resolve_retriever``.
        """
        cfg = self._config
        if not cfg.enabled or not self._card_store.cards:
            return []

        resolver = resolve_retriever_fn or self.resolve_retriever

        query_by_key = build_dedup_queries(card.model_dump())
        tool_by_key = {
            QUERY_DESCRIPTION: "vector_description",
            QUERY_EXPLANATION_SUMMARY: "vector_explanation_summary",
            QUERY_DESCRIPTION_EXPLANATION_SUMMARY: "vector_description_explanation_summary",
            QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY: "vector_description_task_description_summary",
        }

        scores_by_query: dict[str, dict[str, float]] = {}
        cards = self._card_store.cards

        for query_key, query_text in query_by_key.items():
            text = str(query_text or "").strip()
            if not text:
                continue

            retriever = resolver(tool_by_key[query_key])
            if retriever is None:
                continue

            try:
                hits_by_query = retriever.search([text], top_k=cfg.top_k_per_query)
            except Exception as exc:
                logger.warning(
                    "[Memory] Dedup retrieval failed for query '{}': {}",
                    query_key,
                    exc,
                )
                continue

            hits = []
            if isinstance(hits_by_query, list) and hits_by_query:
                first = hits_by_query[0]
                if isinstance(first, list):
                    hits = first
                else:
                    hits = hits_by_query

            query_scores: dict[str, float] = {}
            for hit in hits:
                card_id = str(getattr(hit, "page_id", "") or "").strip()
                if not card_id or card_id not in cards:
                    continue
                if is_program_card(cards[card_id]):
                    continue
                meta = getattr(hit, "meta", {}) or {}
                try:
                    score = float(meta.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                if score <= 0:
                    continue
                previous_score = query_scores.get(card_id, 0.0)
                if score > previous_score:
                    query_scores[card_id] = score
            scores_by_query[query_key] = query_scores

        return compute_weighted_candidates(
            scores_by_query,
            weights=cfg.weights,
            final_top_n=cfg.final_top_n,
            min_final_score=cfg.min_final_score,
        )

    # --- LLM formatting ---

    def format_for_llm(
        self, scored_candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Format scored candidates for the LLM dedup prompt."""
        payload: list[dict[str, Any]] = []
        cards = self._card_store.cards
        for item in scored_candidates:
            card_id = str(item.get("card_id") or "").strip()
            if not card_id:
                continue
            card = cards.get(card_id)
            if card is None:
                continue

            card_dict = card.model_dump()
            explanations = get_full_explanations(card_dict)
            payload.append(
                {
                    "card_id": card_id,
                    "final_score": float(item.get("final_score", 0.0)),
                    "scores": item.get("scores", {}),
                    "task_description_summary": truncate_text(
                        card.task_description_summary, 600
                    ),
                    "description": truncate_text(card.description, 1200),
                    "explanation_summary": truncate_text(
                        get_explanation_summary(card_dict), 600
                    ),
                    "explanation_full": [
                        truncate_text(explanation, 1200) for explanation in explanations
                    ],
                }
            )
        return payload

    # --- LLM decision ---

    def decide_action(
        self,
        incoming_card: AnyCard,
        candidates_for_llm: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ask LLM to decide: add, discard, or update."""
        default_decision: dict[str, Any] = {
            "action": "add",
            "reason": "",
            "duplicate_of": "",
            "updates": [],
        }
        if self.llm_service is None or not candidates_for_llm:
            return default_decision

        candidate_ids = {
            str(item.get("card_id") or "").strip()
            for item in candidates_for_llm
            if str(item.get("card_id") or "").strip()
        }
        if not candidate_ids:
            return default_decision

        incoming_dict = incoming_card.model_dump()
        incoming_payload = {
            "id": str(incoming_card.id or "").strip(),
            "task_description_summary": truncate_text(
                incoming_card.task_description_summary, 600
            ),
            "description": truncate_text(incoming_card.description, 1200),
            "explanation_summary": truncate_text(
                get_explanation_summary(incoming_dict), 600
            ),
            "explanation_full": [
                truncate_text(explanation, 1200)
                for explanation in get_full_explanations(incoming_dict)
            ],
        }
        prompt = (
            "You are a memory-card deduplication and update policy agent.\n"
            "For NEW_CARD choose exactly one action:\n"
            "- add: NEW_CARD is genuinely new and should be saved as a new memory card.\n"
            "- discard: one existing card already represents the same idea.\n"
            "- update: idea exists, but NEW_CARD adds a new task/use-case "
            "and/or new explanation details.\n\n"
            "Return only JSON with this schema:\n"
            "{\n"
            '  "action": "add|discard|update",\n'
            '  "reason": "short reason",\n'
            '  "duplicate_of": "card_id or empty",\n'
            '  "updates": [\n'
            "    {\n"
            '      "card_id": "candidate card id",\n'
            '      "update_task_description": true|false,\n'
            '      "task_description_append": "text to append or empty",\n'
            '      "task_description_summary": "updated summary or empty",\n'
            '      "update_explanation": true|false,\n'
            '      "explanation_append": '
            '"full explanation text to append or empty",\n'
            '      "explanation_summary": "updated summary or empty"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Use add when NEW_CARD is a genuinely new idea "
            "and should become its own card.\n"
            "- Use update when one candidate already contains the same core idea, "
            "but NEW_CARD contributes a new use-case, sharper wording, "
            "extra mechanism detail, or additional explanation "
            "that should be merged into that existing card.\n"
            "- Use discard when one candidate already expresses the same idea "
            "with no meaningful new information.\n"
            "- Do not choose update or discard just because cards share "
            "the same broad task, benchmark, or domain.\n"
            "- Compare the actual idea/mechanism/intervention described "
            "in DESCRIPTION and EXPLANATION.\n"
            "- If the core idea in NEW_CARD is meaningfully different "
            "from every candidate, action must be add.\n"
            "- If action=discard, set duplicate_of to one candidate card_id.\n"
            "- If action=update, include one or more update objects "
            "with candidate card_ids.\n"
            "- Never invent card ids outside the candidate list.\n\n"
            f"NEW_CARD:\n"
            f"{json.dumps(incoming_payload, ensure_ascii=True, indent=2)}\n\n"
            f"CANDIDATE_CARDS:\n"
            f"{json.dumps(candidates_for_llm, ensure_ascii=True, indent=2)}"
        )

        cfg = self._config
        decision = default_decision
        for attempt in range(cfg.llm_max_retries):
            try:
                response_text, _, _, _ = self.llm_service.generate(prompt)
            except Exception as exc:
                logger.warning("[Memory] Dedup LLM decision call failed: {}", exc)
                continue
            parsed = parse_llm_card_decision(
                response_text,
                candidate_ids=candidate_ids,
            )
            if parsed is not None:
                decision = parsed
                break
            logger.warning(
                "[Memory] Dedup LLM returned no valid JSON (attempt {}/{})",
                attempt + 1,
                cfg.llm_max_retries,
            )
        return decision

    # --- Merge computation ---

    def compute_merges(
        self,
        incoming_card: AnyCard,
        updates: list[dict[str, Any]],
    ) -> list[tuple[str, AnyCard]]:
        """Compute merged cards from update actions.

        Returns ``(card_id, merged_card)`` pairs — the caller saves them.
        """
        merges: list[tuple[str, AnyCard]] = []
        seen_ids: set[str] = set()
        incoming_dict = incoming_card.model_dump()
        cards = self._card_store.cards

        for update in updates:
            if not isinstance(update, dict):
                continue
            card_id = str(update.get("card_id") or "").strip()
            if not card_id or card_id in seen_ids:
                continue
            existing_card = cards.get(card_id)
            if existing_card is None:
                continue

            existing_dict = existing_card.model_dump()
            merged_dict = merge_updated_card(existing_dict, incoming_dict, update)
            merged_dict["id"] = card_id
            merged_card = normalize_memory_card(merged_dict)
            seen_ids.add(card_id)
            merges.append((card_id, merged_card))

        return merges
