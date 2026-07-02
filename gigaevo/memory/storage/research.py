"""Agentic retrieval over the vector index: plan → retrieve → reflect.

Two structured-output LLM calls per iteration — a planner that turns the
request into scoped vector queries, and a reflector that either finalizes a
shortlist or asks for more retrieval. At most ``max_iters`` iterations; every
node fails to empty so retrieval can never crash the caller.
"""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.events import MemoryResearchStep, emit_memory_event
from gigaevo.memory.storage.bank import CardBank
from gigaevo.memory.storage.base import ResearchRequest, ResearchResult
from gigaevo.memory.storage.config import EmbedConfig, ResearchConfig
from gigaevo.memory.storage.index import VectorIndex
from gigaevo.prompts import load_prompt

_FIELD_CLIP_CHARS = 1200
_PAYLOAD_CLIP_CHARS = 12000
_KEYWORD_LIMIT = 12
_MAX_FOLLOWUP_QUERIES = 5


class ScopedQuery(BaseModel):
    scope: str = Field(description="One of the available search scopes.")
    query: str = Field(
        description="A short natural-language sentence stating the relevance "
        "signal to retrieve."
    )


class SearchPlan(BaseModel):
    queries: list[ScopedQuery] = Field(
        default_factory=list,
        description="Independent retrieval queries, each against one scope.",
    )


class ShortlistDecision(BaseModel):
    mode: Literal["final", "continue"]
    reasoning: str = Field(
        default="",
        description="Brief grounded justification for the decision.",
    )
    selected_ids: list[str] = Field(
        default_factory=list,
        description='card_ids of the selected candidates (mode "final" only).',
    )
    additional_queries: list[str] = Field(
        default_factory=list,
        description='Follow-up retrieval queries (mode "continue" only).',
    )


class _PlannerState(TypedDict, total=False):
    request: str
    context: str
    scopes_section: str
    messages: list[BaseMessage]
    llm_response: Any
    result: SearchPlan
    metadata: dict


class RetrievalPlannerAgent(LangGraphAgent):
    StateSchema = _PlannerState

    def __init__(self, llm: Any, prompts_dir: str | None = None) -> None:
        self.system_prompt = load_prompt("retrieval_planner", "system", prompts_dir)
        self.user_prompt_template = load_prompt(
            "retrieval_planner", "user", prompts_dir
        )
        super().__init__(llm.with_structured_output(SearchPlan))

    def build_prompt(self, state: _PlannerState) -> _PlannerState:
        context = state.get("context", "")
        context_section = f"CONTEXT:\n{context}\n\n" if context else ""
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=self.user_prompt_template.format(
                    request=state["request"],
                    context_section=context_section,
                    scopes_section=state["scopes_section"],
                )
            ),
        ]
        return state

    def parse_response(self, state: _PlannerState) -> _PlannerState:
        resp = state["llm_response"]
        state["result"] = resp if isinstance(resp, SearchPlan) else SearchPlan(**resp)
        return state

    async def arun(
        self, *, request: str, context: str, scopes_section: str
    ) -> SearchPlan:
        state: _PlannerState = {
            "request": request,
            "context": context,
            "scopes_section": scopes_section,
        }
        final = await self.graph.ainvoke(state)
        return final["result"]


class _ReflectionState(TypedDict, total=False):
    request: str
    candidates: str
    max_cards: int
    messages: list[BaseMessage]
    llm_response: Any
    result: ShortlistDecision
    metadata: dict


class RetrievalReflectionAgent(LangGraphAgent):
    StateSchema = _ReflectionState

    def __init__(self, llm: Any, prompts_dir: str | None = None) -> None:
        self.system_prompt = load_prompt("retrieval_reflection", "system", prompts_dir)
        self.user_prompt_template = load_prompt(
            "retrieval_reflection", "user", prompts_dir
        )
        super().__init__(llm.with_structured_output(ShortlistDecision))

    def build_prompt(self, state: _ReflectionState) -> _ReflectionState:
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=self.user_prompt_template.format(
                    request=state["request"],
                    candidates=state["candidates"],
                    max_cards=state["max_cards"],
                )
            ),
        ]
        return state

    def parse_response(self, state: _ReflectionState) -> _ReflectionState:
        resp = state["llm_response"]
        state["result"] = (
            resp if isinstance(resp, ShortlistDecision) else ShortlistDecision(**resp)
        )
        return state

    async def arun(
        self, *, request: str, candidates: str, max_cards: int
    ) -> ShortlistDecision:
        state: _ReflectionState = {
            "request": request,
            "candidates": candidates,
            "max_cards": max_cards,
        }
        final = await self.graph.ainvoke(state)
        return final["result"]


def _clip(text: str, max_chars: int = _FIELD_CLIP_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16] + "...[truncated]"


def _candidate_payload(card: Card) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "card_id": card.id,
        "kind": card.kind.value,
        "description": _clip(card.description),
        "evidence_summary": _clip(card.explanation_summary),
    }
    if card.category:
        payload["category"] = card.category
    if card.task_description_summary:
        payload["task_description_summary"] = _clip(card.task_description_summary)
    if card.keywords:
        payload["keywords"] = list(card.keywords[:_KEYWORD_LIMIT])
    if card.kind is CardKind.PROGRAM and card.fitness is not None:
        payload["fitness"] = card.fitness
    return payload


def _with_focus(request: str, queries: list[str]) -> str:
    cleaned = [q.strip() for q in queries if q.strip()]
    if not cleaned:
        return request
    lines = [request, "", "Follow-up retrieval focus:"]
    lines.extend(f"{i}. {q}" for i, q in enumerate(cleaned, 1))
    return "\n".join(lines)


class ResearchAgent:
    """The loop composing planner, index, and reflector.

    Aggregates candidates across iterations (a card retrieved once stays a
    candidate), so the reflector always judges the full evidence pool. If no
    iteration finalizes, the answer is empty — a shortlist is only ever an
    explicit reflector decision.
    """

    def __init__(
        self,
        llm: Any,
        bank: CardBank,
        index: VectorIndex,
        embed: EmbedConfig,
        config: ResearchConfig,
        query_scopes: tuple[str, ...],
        prompts_dir: str | None = None,
    ) -> None:
        self._bank = bank
        self._index = index
        self._config = config
        self._scopes = query_scopes
        self._scopes_section = "\n".join(
            f'- "{scope}" — searches card fields: {", ".join(embed.embed_scopes[scope])}'
            for scope in query_scopes
        )
        self._planner = RetrievalPlannerAgent(llm, prompts_dir)
        self._reflector = RetrievalReflectionAgent(llm, prompts_dir)

    async def research(self, request: ResearchRequest) -> ResearchResult:
        candidates: dict[str, Card] = {}
        planner_request = request.query
        for step in range(1, self._config.max_iters + 1):
            started = perf_counter()
            plan = await self._plan(planner_request, request.planning_context)
            queries = [
                q for q in plan.queries if q.scope in self._scopes and q.query.strip()
            ]
            new_ids = self._retrieve(queries, request.exclude_ids, candidates)
            decision = await self._reflect(request.query, candidates)
            emit_memory_event(
                MemoryResearchStep(
                    step=step,
                    scopes=tuple(sorted({q.scope for q in queries})),
                    query_count=len(queries),
                    hit_ids=tuple(new_ids),
                    decision=decision.mode,
                    duration_ms=(perf_counter() - started) * 1000.0,
                )
            )
            if decision.mode == "final":
                selected = [
                    candidates[cid]
                    for cid in dict.fromkeys(decision.selected_ids)
                    if cid in candidates
                ]
                return ResearchResult(
                    cards=tuple(selected[: self._config.max_cards]),
                    summary=decision.reasoning,
                    iterations=step,
                )
            planner_request = _with_focus(
                request.query, decision.additional_queries[:_MAX_FOLLOWUP_QUERIES]
            )
        return ResearchResult(iterations=self._config.max_iters)

    async def _plan(self, request: str, context: str) -> SearchPlan:
        try:
            return await self._planner.arun(
                request=request,
                context=context,
                scopes_section=self._scopes_section,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Research] planner failed; continuing with empty plan"
            )
            return SearchPlan()

    def _retrieve(
        self,
        queries: list[ScopedQuery],
        exclude_ids: frozenset[str],
        candidates: dict[str, Card],
    ) -> list[str]:
        new_ids: list[str] = []
        for scoped in queries:
            try:
                hits = self._index.query(
                    scoped.scope,
                    scoped.query,
                    self._config.top_k(scoped.scope),
                    exclude_ids=exclude_ids,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "[Memory][Research] index query failed on scope {}", scoped.scope
                )
                continue
            for hit in hits:
                if hit.card_id in candidates or hit.card_id in exclude_ids:
                    continue
                card = self._bank.get(hit.card_id)
                if card is None:
                    continue
                candidates[card.id] = card
                new_ids.append(card.id)
        return new_ids

    async def _reflect(
        self, request: str, candidates: dict[str, Card]
    ) -> ShortlistDecision:
        payload = json.dumps(
            [_candidate_payload(c) for c in candidates.values()],
            ensure_ascii=True,
            indent=2,
        )
        try:
            return await self._reflector.arun(
                request=request,
                candidates=_clip(payload, _PAYLOAD_CLIP_CHARS),
                max_cards=self._config.max_cards,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Research] reflection failed; continuing without a decision"
            )
            return ShortlistDecision(mode="continue")
