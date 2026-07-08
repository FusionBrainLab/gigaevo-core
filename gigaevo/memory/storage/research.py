"""Agentic retrieval over the vector index: plan → retrieve → reflect.

Two structured-output LLM calls per iteration — a planner that turns the
request into scoped vector queries, and a reflector that either finalizes a
shortlist or asks for more retrieval. At most ``max_iters`` iterations; every
node fails to empty so retrieval can never crash the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
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
from gigaevo.memory.storage.exclusion import expand_exclude_ids, is_card_excluded
from gigaevo.memory.storage.index import VectorIndex
from gigaevo.prompts import load_prompt

_BRIEF_DESCRIPTION_CHARS = 300
_BRIEF_EVIDENCE_CHARS = 160
_BRIEF_TASK_CHARS = 100
_BRIEF_KEYWORD_LIMIT = 6
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
    step_status: str
    observations: str
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
                    step_status=state["step_status"],
                    observations=state["observations"],
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
        self,
        *,
        request: str,
        candidates: str,
        max_cards: int,
        step_status: str,
        observations: str,
    ) -> ShortlistDecision:
        state: _ReflectionState = {
            "request": request,
            "candidates": candidates,
            "max_cards": max_cards,
            "step_status": step_status,
            "observations": observations,
        }
        final = await self.graph.ainvoke(state)
        return final["result"]


def _clip(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16] + "...[truncated]"


def _brief_text(text: str, max_chars: int) -> str:
    return _clip(" ".join(text.split()), max_chars)


def candidate_brief(card: Card) -> dict[str, Any]:
    brief: dict[str, Any] = {
        "card_id": card.id,
        "kind": card.kind.value,
        "description": _brief_text(card.description, _BRIEF_DESCRIPTION_CHARS),
        "evidence_summary": _brief_text(
            card.explanation_summary, _BRIEF_EVIDENCE_CHARS
        ),
    }
    if card.category:
        brief["category"] = card.category
    if card.task_description_summary:
        brief["task_description_summary"] = _brief_text(
            card.task_description_summary, _BRIEF_TASK_CHARS
        )
    if card.keywords:
        brief["keywords"] = list(card.keywords[:_BRIEF_KEYWORD_LIMIT])
    if card.kind is CardKind.PROGRAM and card.fitness is not None:
        brief["fitness"] = card.fitness
    return brief


def render_candidate_briefs(cards: Sequence[Card], budget: int) -> str:
    rendered, _ = render_candidate_briefs_with_visible_ids(cards, budget)
    return rendered


def render_candidate_briefs_with_visible_ids(
    cards: Sequence[Card], budget: int
) -> tuple[str, frozenset[str]]:
    briefs = [candidate_brief(card) for card in cards]
    omitted_ids: list[str] = []
    while True:
        payload: list[dict[str, Any]] = list(briefs)
        if omitted_ids:
            payload.append({"omitted": len(omitted_ids)})
        rendered = json.dumps(payload, ensure_ascii=True, indent=2)
        if len(rendered) <= budget or len(briefs) <= 1:
            return rendered, frozenset(brief["card_id"] for brief in briefs)
        omitted_ids.insert(0, briefs.pop()["card_id"])


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
    candidate), so the reflector always judges the full evidence pool. If the
    reflector violates the final-step contract and asks to continue after the
    budget is exhausted, the loop falls back to the nearest visible candidates
    instead of discarding a non-empty pool.
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
        self._step_status_template = load_prompt(
            "retrieval_reflection", "step_status", prompts_dir
        )
        self._final_step_snippet = load_prompt(
            "retrieval_reflection", "final_step", prompts_dir
        )
        self._already_held_template = load_prompt(
            "retrieval_reflection", "already_held", prompts_dir
        )
        self._no_new_cards_line = load_prompt(
            "retrieval_reflection", "no_new_cards", prompts_dir
        )

    async def research(self, request: ResearchRequest) -> ResearchResult:
        candidates: dict[str, tuple[Card, float]] = {}
        exclude_ids = expand_exclude_ids(self._bank.snapshot(), request.exclude_ids)
        planner_request = request.query
        held_ids: list[str] = []
        for step in range(1, self._config.max_iters + 1):
            started = perf_counter()
            plan = await self._plan(planner_request, request.planning_context)
            queries = [
                q for q in plan.queries if q.scope in self._scopes and q.query.strip()
            ]
            new_ids = self._retrieve(queries, exclude_ids, candidates)
            decision = await self._reflect(
                request.query,
                candidates,
                step,
                self._observations(step, held_ids, new_ids),
            )
            if decision.mode != "final" and step == self._config.max_iters:
                decision = self._final_step_fallback(candidates, decision)
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
                    candidates[cid][0]
                    for cid in dict.fromkeys(decision.selected_ids)
                    if cid in candidates
                    and not is_card_excluded(candidates[cid][0], exclude_ids)
                ]
                return ResearchResult(
                    cards=tuple(selected[: self._config.max_cards]),
                    summary=decision.reasoning,
                    iterations=step,
                )
            held_ids = [
                cid
                for cid in candidates
                if any(cid in query for query in decision.additional_queries)
            ]
            planner_request = _with_focus(
                request.query, decision.additional_queries[:_MAX_FOLLOWUP_QUERIES]
            )
        return ResearchResult(iterations=self._config.max_iters)

    def _final_step_fallback(
        self,
        candidates: dict[str, tuple[Card, float]],
        decision: ShortlistDecision,
    ) -> ShortlistDecision:
        ordered = [
            card for card, _ in sorted(candidates.values(), key=lambda entry: entry[1])
        ]
        _, visible_ids = render_candidate_briefs_with_visible_ids(
            ordered, self._config.reflect_payload_chars
        )
        selected_ids = [
            card.id
            for card in ordered
            if card.id in visible_ids and not is_card_excluded(card, frozenset())
        ][: self._config.max_cards]
        if selected_ids:
            logger.warning(
                "[Memory][Research] reflector returned continue on final step; "
                "falling back to {} visible candidate(s)",
                len(selected_ids),
            )
        return ShortlistDecision(
            mode="final",
            reasoning=decision.reasoning
            or "Final-step fallback selected the nearest visible candidates.",
            selected_ids=selected_ids,
        )

    def _step_status(self, step: int) -> str:
        status = self._step_status_template.format(
            step=step, max_steps=self._config.max_iters
        )
        if step == self._config.max_iters:
            return f"{status}\n{self._final_step_snippet}"
        return status

    def _observations(self, step: int, held_ids: list[str], new_ids: list[str]) -> str:
        if step == 1:
            return ""
        lines: list[str] = []
        if held_ids:
            lines.append(self._already_held_template.format(ids=", ".join(held_ids)))
        if not new_ids:
            lines.append(self._no_new_cards_line)
        return "".join(f"{line}\n" for line in lines)

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
        candidates: dict[str, tuple[Card, float]],
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
                held = candidates.get(hit.card_id)
                if held is not None:
                    candidates[hit.card_id] = (held[0], min(held[1], hit.distance))
                    continue
                card = self._bank.get(hit.card_id)
                if card is None or is_card_excluded(card, exclude_ids):
                    continue
                candidates[card.id] = (card, hit.distance)
                new_ids.append(card.id)
        return new_ids

    async def _reflect(
        self,
        request: str,
        candidates: dict[str, tuple[Card, float]],
        step: int,
        observations: str,
    ) -> ShortlistDecision:
        ordered = [
            card for card, _ in sorted(candidates.values(), key=lambda entry: entry[1])
        ]
        payload, visible_ids = render_candidate_briefs_with_visible_ids(
            ordered, self._config.reflect_payload_chars
        )
        try:
            decision = await self._reflector.arun(
                request=request,
                candidates=payload,
                max_cards=self._config.max_cards,
                step_status=self._step_status(step),
                observations=observations,
            )
            if decision.mode == "final":
                selected_ids = [
                    cid for cid in decision.selected_ids if cid in visible_ids
                ]
                if selected_ids != decision.selected_ids:
                    return decision.model_copy(update={"selected_ids": selected_ids})
            return decision
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Research] reflection failed; continuing without a decision"
            )
            if step == self._config.max_iters:
                return ShortlistDecision(
                    mode="final",
                    reasoning="Reflection failed on the final retrieval step.",
                )
            return ShortlistDecision(mode="continue")
