# research_agent.py
"""
ResearchAgent Module

This module defines the ResearchAgent for the GAM (General-Agentic-Memory) framework.

- ResearchAgent is responsible for research tasks, reasoning, and advanced information retrieval.
- It interacts with the MemoryAgent to store and access past knowledge as abstracts (memory is represented as a list[str], without events/tags).
- ResearchAgent uses explicit research functions to process queries and generate insights.
- Prompts within the module are placeholders for future extensions, such as customizable instructions or templates.

The module focuses on providing clear abstraction and extensible interfaces for research-related agent functionalities.
"""

from __future__ import annotations

import hashlib
import json
import random
from time import perf_counter
from typing import Any

from loguru import logger

from gigaevo.memory._vendor.GAM_root.gam.generator import AbsGenerator
from gigaevo.memory._vendor.GAM_root.gam.prompts import (
    Decision_PROMPT,
    Planning_PROMPT,
)
from gigaevo.memory._vendor.GAM_root.gam.prompts.research_prompts import (
    render_tool_section,
)
from gigaevo.memory._vendor.GAM_root.gam.schemas import (
    DECISION_SCHEMA,
    PLANNING_SCHEMA,
    Decision,
    Hit,
    InMemoryMemoryStore,
    MemoryState,
    MemoryStore,
    PageStore,
    ResearchOutput,
    Result,
    Retriever,
    SearchPlan,
    ToolRegistry,
    TopIdea,
)
from gigaevo.memory.core.events import emit_memory_event

_VECTOR_TOOLS = {
    "vector",
    "vector_description",
    "vector_task_description",
    "vector_explanation_summary",
}
_DEFAULT_TOP_K_BY_TOOL = {
    "vector": 5,
    "vector_description": 5,
    "vector_task_description": 5,
    "vector_explanation_summary": 5,
    "page_index": 5,
}
_TOOL_ORDER = [
    "vector",
    "vector_description",
    "vector_task_description",
    "vector_explanation_summary",
    "page_index",
]
_GAM_MILLISECONDS_PER_SECOND = 1000.0
_GAM_TIMING_DECIMALS = 3
_GAM_TEXT_HEAD_CHARS = 240
_GAM_CANDIDATE_TEXT_CHARS = 1200
_GAM_CANDIDATE_LIST_LIMIT = 12
_GAM_EVENT_ID_LIMIT = 50
_GAM_EVENT_QUERY_LIMIT = 8


def _elapsed_ms(started: float) -> float:
    return round(
        (perf_counter() - started) * _GAM_MILLISECONDS_PER_SECOND,
        _GAM_TIMING_DECIMALS,
    )


def _head(value: Any, max_chars: int = _GAM_TEXT_HEAD_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16] + "...[truncated]"


def _query_preview(values: list[Any]) -> list[str]:
    return [_head(value) for value in values[:_GAM_EVENT_QUERY_LIMIT]]


def _hit_page_ids(hits: list[Hit]) -> list[str]:
    out: list[str] = []
    for hit in hits:
        page_id = str(hit.page_id or "").strip()
        if page_id:
            out.append(page_id)
        if len(out) >= _GAM_EVENT_ID_LIMIT:
            break
    return out


def _hit_source_counts(hits: list[Hit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hit in hits:
        source = str(hit.source or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _idea_card_ids(ideas: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for idea in ideas:
        card_id = str(idea.get("card_id") or "").strip()
        if card_id:
            ids.append(card_id)
        if len(ids) >= _GAM_EVENT_ID_LIMIT:
            break
    return ids


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _compact_string(value: Any, max_chars: int = _GAM_CANDIDATE_TEXT_CHARS) -> str:
    return _head(_clean_string(value), max_chars=max_chars)


def _compact_list(value: Any, limit: int = _GAM_CANDIDATE_LIST_LIMIT) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _clean_string(item)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _card_efficacy_summary(card: dict[str, Any]) -> str:
    if not isinstance(card, dict):
        return ""
    if card.get("category") == "program" and card.get("fitness") is not None:
        try:
            return f"exemplar fitness {float(card['fitness']):.4f}"
        except (TypeError, ValueError):
            return ""
    return ""


def _drop_random_ideas(
    ideas: list[dict[str, Any]], dose: int, *, seed_basis: str
) -> list[dict[str, Any]]:
    """Drop ``dose`` of the ranked ideas at random, protecting the top-ranked one
    until the dose empties the slate. Seeded from the pool itself (hashlib, not
    salted ``hash()``) so the drop is reproducible across processes."""
    if dose <= 0 or not ideas:
        return ideas
    if dose >= len(ideas):
        return []
    seed = int(hashlib.sha256(seed_basis.encode()).hexdigest()[:16], 16)
    drop = set(random.Random(seed).sample(range(1, len(ideas)), dose))
    return [idea for i, idea in enumerate(ideas) if i not in drop]


class ResearchAgent:
    """
    Public API:
      - research(request, memory_state=None) -> ResearchOutput
    Internal steps:
      - _planning(request, memory_state) -> SearchPlan
      - _search_no_integrate(plan) -> Result  (calls vector/page_index tools)
      - _reflection(request, retrieved_ideas) -> Decision

    Note: Uses MemoryStore to dynamically load current memory state.
    This allows ResearchAgent to access the latest memory updates from MemoryAgent.
    """

    def __init__(
        self,
        page_store: PageStore,
        memory_store: MemoryStore | None = None,
        tool_registry: ToolRegistry | None = None,
        retrievers: dict[str, Retriever] | None = None,
        generator: AbsGenerator | None = None,
        max_iters: int = 3,
        allowed_tools: list[str] | None = None,
        top_k_by_tool: dict[str, int] | None = None,
        dir_path: str | None = None,
        system_prompts: dict[str, str] | None = None,
        max_cards: int = 3,
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for ResearchAgent")
        self.page_store = page_store
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.generator = generator
        self.max_iters = max_iters
        self._allowed_tools = self._normalize_allowed_tools(allowed_tools)
        self._top_k_by_tool = self._normalize_top_k_by_tool(top_k_by_tool)
        self.max_cards = max(1, int(max_cards))

        default_system_prompts = {"planning": "", "integration": "", "reflection": ""}
        if system_prompts is None:
            self.system_prompts = default_system_prompts
        else:
            self.system_prompts = {**default_system_prompts, **system_prompts}

        # Build indices upfront (if retrievers are provided)
        for name, r in self.retrievers.items():
            try:
                r.build(self.page_store)
                logger.debug(
                    "[Memory][GAM][ResearchAgent][Init] Successfully built {} retriever",
                    name,
                )
            except Exception as e:
                logger.error(
                    "[Memory][GAM][ResearchAgent][Init] Failed to build {} retriever: {}",
                    name,
                    e,
                )
                pass

    @staticmethod
    def _normalize_allowed_tools(allowed_tools: list[str] | None) -> set[str]:
        supported_tools = {"page_index", *_VECTOR_TOOLS}
        if not allowed_tools:
            return supported_tools

        normalized = {str(tool).strip() for tool in allowed_tools if str(tool).strip()}
        filtered = {tool for tool in normalized if tool in supported_tools}
        return filtered or supported_tools

    @staticmethod
    def _normalize_top_k_by_tool(
        top_k_by_tool: dict[str, int] | None,
    ) -> dict[str, int]:
        normalized = dict(_DEFAULT_TOP_K_BY_TOOL)
        if not isinstance(top_k_by_tool, dict):
            return normalized

        for tool_name, raw_value in top_k_by_tool.items():
            tool = str(tool_name).strip()
            if tool not in normalized:
                continue
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                normalized[tool] = value
        return normalized

    def _tool_top_k(self, tool: str) -> int:
        return self._top_k_by_tool.get(tool, _DEFAULT_TOP_K_BY_TOOL.get(tool, 5))

    @staticmethod
    def _normalize_query_list(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        cleaned: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if text:
                cleaned.append(text)
        return cleaned

    def _vector_queries_for_tool(self, plan: SearchPlan, tool: str) -> list[str]:
        if tool == "vector":
            return self._normalize_query_list(plan.vector_queries)
        if tool == "vector_description":
            return self._normalize_query_list(
                plan.vector_description_queries
            ) or self._normalize_query_list(plan.vector_queries)
        if tool == "vector_task_description":
            return self._normalize_query_list(
                plan.vector_task_description_queries
            ) or self._normalize_query_list(plan.vector_queries)
        if tool == "vector_explanation_summary":
            return self._normalize_query_list(
                plan.vector_explanation_summary_queries
            ) or self._normalize_query_list(plan.vector_queries)
        return []

    def _active_tools(self) -> list[str]:
        return [
            tool
            for tool in _TOOL_ORDER
            if tool in self._allowed_tools and self._tool_top_k(tool) > 0
        ]

    def _filter_tools(self, tools: list[str]) -> list[str]:
        active = set(self._active_tools())
        return [tool for tool in tools if tool in active]

    def _emit_gam_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        level: str = "DEBUG",
    ) -> None:
        try:
            emit_memory_event(
                component="GAM",
                event_type=event_type,
                payload=payload or {},
                level=level,
            )
        except Exception as exc:
            logger.debug(
                "[Memory][GAM][ResearchAgent][Event] Failed to emit {} event: {}",
                event_type,
                exc,
            )

    def _plan_payload(self, plan: SearchPlan) -> dict[str, Any]:
        vector_query_counts = {
            "vector": len(self._normalize_query_list(plan.vector_queries)),
            "vector_description": len(
                self._normalize_query_list(plan.vector_description_queries)
            ),
            "vector_task_description": len(
                self._normalize_query_list(plan.vector_task_description_queries)
            ),
            "vector_explanation_summary": len(
                self._normalize_query_list(plan.vector_explanation_summary_queries)
            ),
        }
        return {
            "tools": list(plan.tools),
            "filtered_tools": self._filter_tools(plan.tools),
            "active_tools": self._active_tools(),
            "keyword_count": len(self._normalize_query_list(plan.keyword_collection)),
            "vector_query_counts": vector_query_counts,
            "page_index_count": len(plan.page_index or []),
            "keyword_preview": _query_preview(plan.keyword_collection),
            "vector_preview": {
                "vector": _query_preview(plan.vector_queries),
                "vector_description": _query_preview(plan.vector_description_queries),
                "vector_task_description": _query_preview(
                    plan.vector_task_description_queries
                ),
                "vector_explanation_summary": _query_preview(
                    plan.vector_explanation_summary_queries
                ),
            },
            "page_index_preview": list(
                (plan.page_index or [])[:_GAM_EVENT_QUERY_LIMIT]
            ),
        }

    def _emit_tool_search_event(
        self,
        *,
        mode: str,
        tool: str,
        queries: list[Any],
        top_k: int,
        hits: list[Hit],
    ) -> None:
        self._emit_gam_event(
            "gam.search.tool",
            {
                "mode": mode,
                "tool": tool,
                "top_k": top_k,
                "query_count": len(queries),
                "query_preview": _query_preview(queries),
                "hit_count": len(hits),
                "hit_ids": _hit_page_ids(hits),
                "hit_sources": _hit_source_counts(hits),
            },
        )

    # ---- Public ----
    def research(
        self,
        request: str,
        memory_state: str | None = None,
        planning_request: str | None = None,
        *,
        exclude_ids: frozenset[str] = frozenset(),
        random_drop_dose: int = 0,
    ) -> ResearchOutput:
        started = perf_counter()
        base_payload = {
            "request_chars": len(request or ""),
            "planning_request_chars": len(planning_request or ""),
            "memory_state_chars": len(memory_state or ""),
            "max_iters": self.max_iters,
            "max_cards": self.max_cards,
            "active_tools": self._active_tools(),
            "top_k_by_tool": dict(self._top_k_by_tool),
            "exclude_count": len(exclude_ids),
            "exclude_ids": sorted(exclude_ids),
            "random_drop_dose": random_drop_dose,
        }
        self._emit_gam_event("gam.research.start", base_payload)
        try:
            self._update_retrievers()

            output = self._research(
                request,
                memory_state=memory_state,
                planning_request=planning_request,
                exclude_ids=exclude_ids,
                random_drop_dose=random_drop_dose,
            )
            self._emit_gam_event(
                "gam.research.complete",
                {
                    **base_payload,
                    "outcome": "ok",
                    "integrated_memory_chars": len(output.integrated_memory or ""),
                    "raw_memory_type": type(output.raw_memory).__name__,
                    "duration_ms": _elapsed_ms(started),
                },
            )
            return output
        except Exception as exc:
            self._emit_gam_event(
                "gam.research.exception",
                {
                    **base_payload,
                    "outcome": "exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "duration_ms": _elapsed_ms(started),
                },
                level="WARNING",
            )
            raise

    def _research(
        self,
        request: str,
        memory_state: str | None = None,
        planning_request: str | None = None,
        *,
        exclude_ids: frozenset[str] = frozenset(),
        random_drop_dose: int = 0,
    ) -> ResearchOutput:
        iterations: list[dict[str, Any]] = []
        planning_base = planning_request or request
        next_request = planning_base
        retrieved_ideas_by_id: dict[str, dict[str, Any]] = {}
        final_decision: Decision | None = None

        for step in range(self.max_iters):
            step_started = perf_counter()
            current_memory_state = self.memory_store.load()
            plan = self._planning(
                next_request,
                current_memory_state,
                memory_state_override=memory_state,
            )
            plan.tools = self._filter_tools(plan.tools)
            logger.debug(
                "[Memory][GAM][ResearchAgent][Planning] Plan: {}",
                json.dumps(plan.__dict__, ensure_ascii=True, indent=2),
            )

            retrieved = self._search_no_integrate(
                plan,
                Result(),
                request,
                exclude_ids=exclude_ids,
                random_drop_dose=random_drop_dose,
            )
            iteration_ideas = self._parse_retrieved_ideas(retrieved.content)
            for idea in iteration_ideas:
                card_id = str(idea.get("card_id") or "").strip()
                if not card_id:
                    continue
                if card_id not in retrieved_ideas_by_id:
                    retrieved_ideas_by_id[card_id] = idea

            aggregated_ideas = list(retrieved_ideas_by_id.values())
            decision = self._reflection(
                request=request,
                retrieved_ideas=aggregated_ideas,
            )
            logger.debug(
                "[Memory][GAM][ResearchAgent][Reflection] Decision: {}",
                json.dumps(decision.model_dump(), ensure_ascii=True, indent=2),
            )

            self._emit_gam_event(
                "gam.iteration",
                {
                    "step": step,
                    "max_iters": self.max_iters,
                    "plan_tools": list(plan.tools),
                    "filtered_tools": self._filter_tools(plan.tools),
                    "retrieved_idea_count": len(iteration_ideas),
                    "aggregated_idea_count": len(aggregated_ideas),
                    "decision_mode": decision.mode,
                    "top_idea_ids": [idea.card_id for idea in decision.top_ideas],
                    "additional_query_count": len(decision.additional_queries),
                    "duration_ms": _elapsed_ms(step_started),
                },
            )
            iterations.append(
                {
                    "step": step,
                    "plan": plan.__dict__,
                    "retrieved": retrieved.__dict__,
                    "retrieved_ideas": iteration_ideas,
                    "decision": decision.model_dump(),
                }
            )

            if decision.mode == "final":
                final_decision = decision
                break

            next_request = self._next_request_from_queries(
                original_request=planning_base,
                queries=decision.additional_queries,
            )

        if final_decision is None:
            final_decision = Decision(
                mode="final",
                top_ideas=[],
                additional_queries=[],
            )

        final_output = self._format_top_ideas(final_decision.top_ideas)
        raw = {
            "iterations": iterations,
            "final_decision": final_decision.model_dump(),
            "retrieved_ideas": list(retrieved_ideas_by_id.values()),
            "evidence_sources": list(retrieved_ideas_by_id.keys()),
        }
        return ResearchOutput(integrated_memory=final_output, raw_memory=raw)

    @staticmethod
    def _next_request_from_queries(original_request: str, queries: list[str]) -> str:
        cleaned = [q for q in queries if str(q or "").strip()]
        if not cleaned:
            return original_request
        lines = [original_request, "", "Follow-up retrieval focus:"]
        for idx, query in enumerate(cleaned, 1):
            lines.append(f"{idx}. {query}")
        return "\n".join(lines)

    @staticmethod
    def _truncate_text(value: str, max_chars: int = 12000) -> str:
        text = str(value or "")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 16] + "\n...[truncated]"

    @staticmethod
    def _as_string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value or "").strip()
        return [text] if text else []

    def _card_map_by_id(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for page in self.page_store.load():
            meta = getattr(page, "meta", None)
            if not isinstance(meta, dict):
                continue
            card = meta.get("amem")
            card_id = str(meta.get("amem_id") or "").strip()
            if isinstance(card, dict):
                card_id = str(card.get("id") or card_id).strip()
                if card_id and card_id not in out:
                    out[card_id] = card
                continue
            if card_id and card_id not in out:
                out[card_id] = {
                    "id": card_id,
                    "description": str(getattr(page, "content", "") or ""),
                }
        return out

    @staticmethod
    def _extract_explanation_summary(card: dict[str, Any]) -> str:
        top_level = str(card.get("explanation_summary") or "").strip()
        if top_level:
            return top_level
        explanation = card.get("explanation")
        if isinstance(explanation, dict):
            return str(explanation.get("summary") or "").strip()
        return ""

    def _build_retrieved_ideas(
        self,
        hits: list[Hit],
        *,
        exclude_ids: frozenset[str] = frozenset(),
        random_drop_dose: int = 0,
    ) -> list[dict[str, Any]]:
        card_map = self._card_map_by_id()
        ideas: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for hit in hits:
            card_id = str(hit.page_id or "").strip()
            if not card_id or card_id in seen_ids:
                continue
            if card_id in exclude_ids:
                continue
            seen_ids.add(card_id)

            card = card_map.get(card_id, {})
            description = _compact_string(card.get("description"))
            if not description:
                description = _compact_string(hit.snippet)

            evidence_summary = self._extract_explanation_summary(card)
            if not evidence_summary:
                evidence_summary = _compact_string(hit.snippet)

            idea: dict[str, Any] = {
                "card_id": card_id,
                "description": description,
                "evidence_summary": _compact_string(evidence_summary),
            }
            for source_key, target_key in (
                ("task_description_summary", "task_description_summary"),
                ("task_description", "task_description"),
                ("strategy", "strategy"),
                ("category", "category"),
            ):
                text = _compact_string(card.get(source_key))
                if text:
                    idea[target_key] = text

            keywords = _compact_list(card.get("keywords"))
            if keywords:
                idea["keywords"] = keywords
            works_with = _compact_list(card.get("works_with"))
            if works_with:
                idea["works_with"] = works_with
            links = _compact_list(card.get("links"))
            if links:
                idea["links"] = links
            efficacy = _card_efficacy_summary(card)
            if efficacy:
                idea["efficacy"] = efficacy
            source = str(hit.source or "").strip()
            if source:
                idea["evidence_source"] = source
            score = hit.meta.get("score") if isinstance(hit.meta, dict) else None
            if isinstance(score, (int, float)):
                idea["score"] = float(score)
            ideas.append(idea)

        if random_drop_dose > 0:
            seed_basis = "|".join(i["card_id"] for i in ideas) + f"#{random_drop_dose}"
            ideas = _drop_random_ideas(ideas, random_drop_dose, seed_basis=seed_basis)
        return ideas

    def _parse_retrieved_ideas(self, payload: Any) -> list[dict[str, Any]]:
        raw = payload
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return []
            try:
                raw = json.loads(text)
            except Exception:
                return []

        if not isinstance(raw, list):
            return []

        ideas: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            card_id = str(item.get("card_id") or "").strip()
            if not card_id or card_id in seen_ids:
                continue
            seen_ids.add(card_id)

            idea: dict[str, Any] = {
                "card_id": card_id,
                "description": _compact_string(item.get("description")),
                "evidence_summary": _compact_string(item.get("evidence_summary")),
            }
            for key in (
                "task_description_summary",
                "task_description",
                "strategy",
                "category",
                "efficacy",
            ):
                text = _compact_string(item.get(key))
                if text:
                    idea[key] = text
            for key in ("keywords", "works_with", "links"):
                values = _compact_list(item.get(key))
                if values:
                    idea[key] = values
            evidence_source = str(item.get("evidence_source") or "").strip()
            if evidence_source:
                idea["evidence_source"] = evidence_source
            score = item.get("score")
            if isinstance(score, (int, float)):
                idea["score"] = float(score)
            ideas.append(idea)
        return ideas

    def _reflection(
        self,
        request: str,
        retrieved_ideas: list[dict[str, Any]],
    ) -> Decision:
        started = perf_counter()
        normalized_ideas = self._parse_retrieved_ideas(retrieved_ideas)
        card_ids = [str(item.get("card_id") or "").strip() for item in normalized_ideas]
        ideas_payload = self._truncate_text(
            json.dumps(normalized_ideas, ensure_ascii=True, indent=2)
            if normalized_ideas
            else "[]"
        )

        system_prompt = self.system_prompts.get("reflection")
        template_prompt = Decision_PROMPT.format(
            request=request,
            retrieved_ideas=ideas_payload,
            max_cards=self.max_cards,
        )
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        response: dict[str, Any] | None = None
        try:
            response = self.generator.generate_single(
                prompt=prompt, schema=DECISION_SCHEMA
            )
            data = response.get("json") or json.loads(response["text"])
        except Exception as e:
            if isinstance(response, dict):
                text_head = str(response.get("text") or "")[:200]
                response_ctx = (
                    f"response_keys={sorted(response)} text_head={text_head!r}"
                )
            else:
                response_ctx = f"response={response!r} (LLM call itself failed)"
            self._emit_gam_event(
                "gam.reflection",
                {
                    "outcome": "exception",
                    "mode": "continue",
                    "prompt_chars": len(prompt),
                    "request_chars": len(request or ""),
                    "retrieved_idea_count": len(normalized_ideas),
                    "candidate_ids": card_ids[:_GAM_EVENT_ID_LIMIT],
                    "top_idea_ids": [],
                    "additional_query_count": 0,
                    "response_context": _head(response_ctx),
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "duration_ms": _elapsed_ms(started),
                },
                level="WARNING",
            )
            logger.opt(exception=True).error(
                "[Memory][GAM][ResearchAgent][Reflection] LLM call "
                "failed; falling back to mode=continue with 0 ideas kept | "
                "{}: {} | schema=Decision prompt_chars={} "
                "retrieved_ideas={} | {}",
                type(e).__name__,
                e,
                len(prompt),
                len(normalized_ideas),
                response_ctx,
            )
            return Decision(
                mode="continue", top_ideas=[], additional_queries=[]
            )

        raw_mode = str((data or {}).get("mode") or "continue").strip().lower()
        mode = "final" if raw_mode == "final" else "continue"

        additional_queries = self._normalize_query_list(
            (data or {}).get("additional_queries")
        )
        raw_ideas = (data or {}).get("top_ideas")
        top_ideas: list[TopIdea] = []
        seen_ids: set[str] = set()

        if isinstance(raw_ideas, list):
            for item in raw_ideas:
                if isinstance(item, str):
                    item = {"card_id": item}
                if not isinstance(item, dict):
                    continue
                card_id = str(item.get("card_id") or "").strip()
                if not card_id or card_id in seen_ids:
                    continue
                if not card_ids or card_id not in card_ids:
                    continue
                seen_ids.add(card_id)
                top_ideas.append(
                    TopIdea(
                        card_id=card_id,
                    )
                )

        if mode == "final":
            decision = Decision(
                mode="final",
                top_ideas=top_ideas[: self.max_cards],
                additional_queries=[],
            )
            self._emit_gam_event(
                "gam.reflection",
                {
                    "outcome": "ok",
                    "mode": "final",
                    "prompt_chars": len(prompt),
                    "request_chars": len(request or ""),
                    "retrieved_idea_count": len(normalized_ideas),
                    "candidate_ids": card_ids[:_GAM_EVENT_ID_LIMIT],
                    "top_idea_ids": [idea.card_id for idea in decision.top_ideas],
                    "additional_query_count": 0,
                    "duration_ms": _elapsed_ms(started),
                },
            )
            return decision
        decision = Decision(
            mode="continue", top_ideas=[], additional_queries=additional_queries[:5]
        )
        self._emit_gam_event(
            "gam.reflection",
            {
                "outcome": "ok",
                "mode": "continue",
                "prompt_chars": len(prompt),
                "request_chars": len(request or ""),
                "retrieved_idea_count": len(normalized_ideas),
                "candidate_ids": card_ids[:_GAM_EVENT_ID_LIMIT],
                "top_idea_ids": [],
                "additional_query_count": len(decision.additional_queries),
                "additional_query_preview": _query_preview(decision.additional_queries),
                "duration_ms": _elapsed_ms(started),
            },
        )
        return decision

    def _format_top_ideas(self, top_ideas: list[TopIdea]) -> str:
        if not top_ideas:
            return "No final top ideas available from experimental pipeline."

        card_map = self._card_map_by_id()
        lines = ["Top selected memory ideas (experimental):", ""]
        for idx, idea in enumerate(top_ideas, 1):
            card = card_map.get(idea.card_id, {})
            description = str(card.get("description") or "").strip()
            evidence_summary = self._extract_explanation_summary(card)
            lines.append(
                f"{idx}. DESCRIPTION: {description or '(not provided in original card)'}"
            )
            lines.append("WHEN_TO_USE:")
            lines.append(
                f"- {evidence_summary or '(not provided in evidence summary)'}"
            )
            lines.append("")
        return "\n".join(lines).strip()

    def _update_retrievers(self):
        """Keep retriever indices in sync with the page store."""
        started = perf_counter()
        current_page_count = len(self.page_store.load())
        previous_page_count = getattr(self, "_last_page_count", None)
        changed = (
            previous_page_count is not None
            and current_page_count != previous_page_count
        )
        updated: list[str] = []
        failed: list[dict[str, str]] = []

        if changed:
            logger.debug(
                "[Memory][GAM][ResearchAgent][RetrieverUpdate] Page count changed "
                "({} -> {}); updating retriever indices.",
                previous_page_count,
                current_page_count,
            )
            for name, retriever in self.retrievers.items():
                try:
                    retriever.update(self.page_store)
                    updated.append(name)
                    logger.debug(
                        "[Memory][GAM][ResearchAgent][RetrieverUpdate] Updated {} "
                        "retriever index",
                        name,
                    )
                except Exception as e:
                    failed.append(
                        {
                            "name": name,
                            "error_type": type(e).__name__,
                            "error": str(e),
                        }
                    )
                    logger.error(
                        "[Memory][GAM][ResearchAgent][RetrieverUpdate] Failed to "
                        "update {} retriever: {}",
                        name,
                        e,
                    )

        self._last_page_count = current_page_count
        self._emit_gam_event(
            "gam.retriever_update",
            {
                "outcome": "updated" if changed and updated else "unchanged",
                "changed": changed,
                "previous_page_count": previous_page_count,
                "current_page_count": current_page_count,
                "retriever_count": len(self.retrievers),
                "updated": updated,
                "failed": failed,
                "duration_ms": _elapsed_ms(started),
            },
            level="WARNING" if failed else "DEBUG",
        )

    # ---- Internal ----
    def _planning(
        self,
        request: str,
        memory_state: MemoryState,
        planning_prompt: str | None = None,
        memory_state_override: str | None = None,
    ) -> SearchPlan:
        """
        Produce a SearchPlan:
          - what specific info is needed
          - which tools are useful + inputs
          - vector/page_index payloads
        """
        started = perf_counter()

        if memory_state_override is not None:
            memory_context = memory_state_override
        elif not memory_state.abstracts:
            memory_context = "No memory currently."
        else:
            memory_context_lines = []
            for i, abstract in enumerate(memory_state.abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)

        active_tools = self._active_tools()
        system_prompt = self.system_prompts.get("planning")
        template_prompt = Planning_PROMPT.format(
            request=request,
            memory=memory_context,
            tool_names=",".join(f'"{tool}"' for tool in active_tools),
            tool_section=render_tool_section(active_tools),
        )
        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        prompt_chars = len(prompt)
        estimated_tokens = prompt_chars // 4
        logger.debug(
            "[Memory][GAM][ResearchAgent][Planning] Prompt length: {} chars (~{} tokens)",
            prompt_chars,
            estimated_tokens,
        )

        try:
            response = self.generator.generate_single(
                prompt=prompt, schema=PLANNING_SCHEMA
            )
            data = response.get("json") or json.loads(response["text"])
            plan = SearchPlan(
                tools=data.get("tools", []),
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                vector_description_queries=data.get("vector_description_queries", []),
                vector_task_description_queries=data.get(
                    "vector_task_description_queries", []
                ),
                vector_explanation_summary_queries=data.get(
                    "vector_explanation_summary_queries", []
                ),
                page_index=data.get("page_index", []),
            )
            self._emit_gam_event(
                "gam.plan",
                {
                    "outcome": "ok",
                    "request_chars": len(request or ""),
                    "memory_context_chars": len(memory_context or ""),
                    "prompt_chars": prompt_chars,
                    "estimated_tokens": estimated_tokens,
                    "duration_ms": _elapsed_ms(started),
                    **self._plan_payload(plan),
                },
            )
            return plan
        except Exception as e:
            logger.error(
                "[Memory][GAM][ResearchAgent][Planning] Planning failed: {}", e
            )
            plan = SearchPlan(
                tools=[],
                keyword_collection=[],
                vector_queries=[],
                vector_description_queries=[],
                vector_task_description_queries=[],
                vector_explanation_summary_queries=[],
                page_index=[],
            )
            self._emit_gam_event(
                "gam.plan",
                {
                    "outcome": "exception",
                    "request_chars": len(request or ""),
                    "memory_context_chars": len(memory_context or ""),
                    "prompt_chars": prompt_chars,
                    "estimated_tokens": estimated_tokens,
                    "duration_ms": _elapsed_ms(started),
                    "error_type": type(e).__name__,
                    "error": str(e),
                    **self._plan_payload(plan),
                },
                level="WARNING",
            )
            return plan

    def _search_no_integrate(
        self,
        plan: SearchPlan,
        result: Result,
        question: str,
        *,
        exclude_ids: frozenset[str] = frozenset(),
        random_drop_dose: int = 0,
    ) -> Result:
        """
        Search without integration:
          1) Execute search tools
          2) Collect all hits without LLM integration
          3) Format hits as plain text results
        Returns Result with raw search hits formatted as content.
        """
        started = perf_counter()
        all_hits: list[Hit] = []
        selected_tools = self._filter_tools(plan.tools)

        # Execute each planned tool and collect hits
        for tool in selected_tools:
            hits: list[Hit] = []
            tool_queries: list[Any] = []
            tool_top_k = self._tool_top_k(tool)

            if tool in _VECTOR_TOOLS:
                vector_queries = self._vector_queries_for_tool(plan, tool)
                tool_queries = list(vector_queries)
                if vector_queries:
                    vector_results = self._search_by_vector_tool(
                        tool_name=tool,
                        query_list=vector_queries,
                        top_k=tool_top_k,
                    )
                    # Flatten the results if they come as List[List[Hit]]
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)

            elif tool == "page_index":
                tool_queries = list(plan.page_index or [])
                if plan.page_index:
                    target_page_index = plan.page_index[:tool_top_k]
                    page_results = self._search_by_page_index(target_page_index)
                    # Flatten the results if they come as List[List[Hit]]
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)

            self._emit_tool_search_event(
                mode="no_integrate",
                tool=tool,
                queries=tool_queries,
                top_k=tool_top_k,
                hits=hits,
            )

        # Format all hits as text content without integration
        if not all_hits:
            self._emit_gam_event(
                "gam.search",
                {
                    "mode": "no_integrate",
                    "outcome": "no_hits",
                    "selected_tools": selected_tools,
                    "raw_hit_count": 0,
                    "unique_hit_count": 0,
                    "exclude_count": len(exclude_ids),
                    "exclude_ids": sorted(exclude_ids),
                    "random_drop_dose": random_drop_dose,
                    "duration_ms": _elapsed_ms(started),
                },
            )
            return result

        # Deduplicate by page_id so cross-tool matches do not repeat evidence.
        unique_hits: dict[str, Hit] = {}
        hits_without_id: list[Hit] = []
        for hit in all_hits:
            if hit.page_id:
                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:
                    existing_hit = unique_hits[hit.page_id]
                    existing_score = (
                        existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    )
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                hits_without_id.append(hit)

        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(
            all_unique_hits,
            key=lambda h: h.meta.get("score", 0) if h.meta else 0,
            reverse=True,
        )

        ideas = self._build_retrieved_ideas(
            sorted_hits, exclude_ids=exclude_ids, random_drop_dose=random_drop_dose
        )
        if not ideas:
            self._emit_gam_event(
                "gam.search",
                {
                    "mode": "no_integrate",
                    "outcome": "no_ideas",
                    "selected_tools": selected_tools,
                    "raw_hit_count": len(all_hits),
                    "unique_hit_count": len(sorted_hits),
                    "hit_ids": _hit_page_ids(sorted_hits),
                    "hit_sources": _hit_source_counts(sorted_hits),
                    "exclude_count": len(exclude_ids),
                    "exclude_ids": sorted(exclude_ids),
                    "random_drop_dose": random_drop_dose,
                    "idea_count": 0,
                    "duration_ms": _elapsed_ms(started),
                },
            )
            return result
        sources = [
            str(item.get("card_id") or "").strip()
            for item in ideas
            if str(item.get("card_id") or "").strip()
        ]
        formatted_content = json.dumps(ideas, ensure_ascii=True, indent=2)

        self._emit_gam_event(
            "gam.search",
            {
                "mode": "no_integrate",
                "outcome": "ideas",
                "selected_tools": selected_tools,
                "raw_hit_count": len(all_hits),
                "unique_hit_count": len(sorted_hits),
                "hit_ids": _hit_page_ids(sorted_hits),
                "hit_sources": _hit_source_counts(sorted_hits),
                "exclude_count": len(exclude_ids),
                "exclude_ids": sorted(exclude_ids),
                "random_drop_dose": random_drop_dose,
                "idea_count": len(ideas),
                "idea_ids": sources[:_GAM_EVENT_ID_LIMIT],
                "content_chars": len(formatted_content),
                "duration_ms": _elapsed_ms(started),
            },
        )
        return Result(
            content=formatted_content if formatted_content else result.content,
            sources=sources if sources else result.sources,
        )

    # ---- search channels ----
    def _search_by_vector(
        self, query_list: list[str], top_k: int = 3
    ) -> list[list[Hit]]:
        return self._search_by_vector_tool("vector", query_list, top_k=top_k)

    def _search_by_vector_tool(
        self,
        tool_name: str,
        query_list: list[str],
        top_k: int = 3,
    ) -> list[list[Hit]]:
        r = self.retrievers.get(tool_name)
        if r is None and tool_name != "vector":
            r = self.retrievers.get("vector")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                logger.error(
                    "[Memory][GAM][ResearchAgent][VectorSearch] Vector search failed "
                    "for {}: {}",
                    tool_name,
                    e,
                )
                return []
        # fallback: none
        return []

    def _search_by_page_index(self, page_index: list[int]) -> list[list[Hit]]:
        r = self.retrievers.get("page_index")
        if r is not None:
            try:
                query_string = ",".join([str(idx) for idx in page_index])
                hits = r.search([query_string], top_k=len(page_index))
                return hits if hits else []
            except Exception as e:
                logger.error(
                    "[Memory][GAM][ResearchAgent][PageIndexSearch] Page index search "
                    "failed: {}",
                    e,
                )
                return []

        out: list[Hit] = []
        for idx in page_index:
            p = self.page_store.get(idx)
            if p:
                amem_id = str((getattr(p, "meta", None) or {}).get("amem_id") or "")
                out.append(
                    Hit(
                        page_id=amem_id.strip() or str(idx),
                        snippet=p.content,
                        source="page_index",
                        meta={"page_index": idx},
                    )
                )
        return [out]

