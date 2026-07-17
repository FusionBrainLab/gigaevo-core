"""Agentic retrieval over the vector index: plan → retrieve → reflect.

Two structured-output LLM calls per iteration — a planner that turns the
request into scoped vector queries, and a reflector that either finalizes a
shortlist or asks for more retrieval. At most ``max_iters`` iterations; every
node fails to empty so retrieval can never crash the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from time import perf_counter
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.memory.cards import Card
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
_MAX_FOLLOWUP_QUERIES = 5
_MAX_PLAN_QUERIES = 5
_RRF_K = 60


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
    """Render only semantic evidence the reflector may use for applicability."""

    brief: dict[str, Any] = {
        "card_id": card.id,
        "kind": card.kind.value,
    }
    brief["description"] = _brief_text(card.description, _BRIEF_DESCRIPTION_CHARS)
    brief["evidence_summary"] = _brief_text(
        card.explanation_summary, _BRIEF_EVIDENCE_CHARS
    )
    if card.task_description_summary:
        brief["task_description_summary"] = _brief_text(
            card.task_description_summary, _BRIEF_TASK_CHARS
        )
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
        self._embed = embed
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
        self._policy_digest = _stable_digest(
            {
                "agent": f"{type(self).__module__}.{type(self).__qualname__}",
                "embed": embed.model_dump(mode="json"),
                "research": config.model_dump(mode="json"),
                "query_scopes": query_scopes,
                "planner_system": self._planner.system_prompt,
                "planner_user": self._planner.user_prompt_template,
                "reflector_system": self._reflector.system_prompt,
                "reflector_user": self._reflector.user_prompt_template,
                "reflection_auxiliary_prompts": {
                    "step_status": self._step_status_template,
                    "final_step": self._final_step_snippet,
                    "already_held": self._already_held_template,
                    "no_new_cards": self._no_new_cards_line,
                },
                "models": tuple(getattr(llm, "model_names", ()) or ()),
            }
        )

    @property
    def policy_digest(self) -> str:
        """Fingerprint the semantic retrieval policy frozen in a decision."""

        return self._policy_digest

    async def research(self, request: ResearchRequest) -> ResearchResult:
        candidates: dict[str, tuple[Card, float]] = {}
        exclude_ids = expand_exclude_ids(self._bank.snapshot(), request.exclude_ids)
        planner_request = request.query
        held_ids: list[str] = []
        for step in range(1, self._config.max_iters + 1):
            started = perf_counter()
            plan = await self._plan(planner_request, request.planning_context)
            queries = self._usable_plan_queries(plan)
            new_ids = self._retrieve(queries, exclude_ids, candidates)
            decision = await self._reflect(
                request.query,
                candidates,
                step,
                self._observations(step, held_ids, new_ids),
            )
            if decision.mode != "final" and step == self._config.max_iters:
                decision = self._final_step_fallback(
                    request.query, candidates, decision
                )
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
        request: str,
        candidates: dict[str, tuple[Card, float]],
        decision: ShortlistDecision,
    ) -> ShortlistDecision:
        ordered = self._ordered_candidates(request, candidates)
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

    def _usable_plan_queries(self, plan: SearchPlan) -> list[ScopedQuery]:
        """Bound and deduplicate planner output before it reaches the index."""

        queries: list[ScopedQuery] = []
        seen: set[tuple[str, str]] = set()
        for scoped in plan.queries:
            normalized = " ".join(scoped.query.split())
            key = (scoped.scope, normalized.casefold())
            if scoped.scope not in self._scopes or not normalized or key in seen:
                continue
            seen.add(key)
            queries.append(scoped.model_copy(update={"query": normalized}))
            if len(queries) == _MAX_PLAN_QUERIES:
                break
        return queries

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
            for rank, hit in enumerate(hits, start=1):
                # Ranks are comparable across scopes and query formulations;
                # raw embedding distances are not. Reciprocal-rank fusion also
                # rewards cards independently retrieved by several signals.
                score = 1.0 / (_RRF_K + rank)
                held = candidates.get(hit.card_id)
                if held is not None:
                    candidates[hit.card_id] = (held[0], held[1] + score)
                    continue
                card = self._bank.get(hit.card_id)
                if card is None or is_card_excluded(card, exclude_ids):
                    continue
                candidates[card.id] = (card, score)
                new_ids.append(card.id)
        return new_ids

    def _ordered_candidates(
        self,
        request: str,
        candidates: dict[str, tuple[Card, float]],
    ) -> list[Card]:
        """Apply RRF relevance, then optional MMR diversity, deterministically."""

        ranked_ids = [
            card_id
            for card_id, _ in sorted(
                candidates.items(), key=lambda item: (-item[1][1], item[0])
            )
        ]
        if not ranked_ids:
            return []
        scores = {card_id: candidates[card_id][1] for card_id in ranked_ids}
        lower, upper = min(scores.values()), max(scores.values())
        relevance = (
            {card_id: 1.0 for card_id in ranked_ids}
            if math.isclose(lower, upper)
            else {
                card_id: (score - lower) / (upper - lower)
                for card_id, score in scores.items()
            }
        )
        try:
            ranked_ids = self._index.mmr_order(
                self._embed.nearest_scope,
                request,
                ranked_ids,
                lambda_=self._config.mmr_lambda,
                relevance=relevance,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Research] MMR ordering failed; using RRF relevance order"
            )
        return [candidates[card_id][0] for card_id in ranked_ids]

    async def _reflect(
        self,
        request: str,
        candidates: dict[str, tuple[Card, float]],
        step: int,
        observations: str,
    ) -> ShortlistDecision:
        ordered = self._ordered_candidates(request, candidates)
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
                    cid
                    for cid in dict.fromkeys(decision.selected_ids)
                    if cid in visible_ids
                ][: self._config.max_cards]
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


def _stable_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
