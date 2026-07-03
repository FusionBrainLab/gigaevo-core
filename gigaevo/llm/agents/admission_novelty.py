"""Novelty-admission agent: a single LangGraph LLM hop gating idea-card writes.

Given one authored insight card and the task baked into its system prompt, the
judge answers a single question: would a strong optimizer LLM already reach for
this lever unprompted on this task? If yes, the card restates the mutator's own
prior and is inert — it is rejected before it enters the bank. If no, it carries
a lever the mutator would not otherwise apply, and it is kept.

This is a NOVELTY-vs-prior test, not a quality or correctness test: a card can be
sound advice yet be rejected for being obvious, and a non-obvious card is kept
even if it is wrong (the mutator's own prior corrects a stray wrong card in
practice). Structured output is bound at construction via
``with_structured_output`` so the router negotiates the method (do not force
``function_calling``).
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter


class NoveltyVerdict(BaseModel):
    keep: bool = Field(
        description="True only if a strong optimizer LLM would NOT already reach "
        "for this lever unprompted on this task — i.e. the card carries something "
        "beyond the model's own prior. False for textbook metaheuristic boilerplate "
        "the model emits cold."
    )
    reason: str = Field(
        default="",
        description="One short sentence justifying the verdict — what the model "
        "already knows (reject) or what the card adds beyond the prior (keep).",
    )


class NoveltyAdmissionState(TypedDict, total=False):
    description: str
    explanation_summary: str
    messages: list[BaseMessage]
    llm_response: Any
    result: NoveltyVerdict
    metadata: dict


class NoveltyAdmissionAgent(LangGraphAgent):
    StateSchema = NoveltyAdmissionState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        super().__init__(llm.with_structured_output(NoveltyVerdict))

    def build_prompt(self, state: NoveltyAdmissionState) -> NoveltyAdmissionState:
        user = self.user_prompt_template.format(
            description=state["description"],
            explanation_summary=state.get("explanation_summary") or "(none)",
        )
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user),
        ]
        return state

    def parse_response(self, state: NoveltyAdmissionState) -> NoveltyAdmissionState:
        resp = state["llm_response"]
        state["result"] = (
            resp if isinstance(resp, NoveltyVerdict) else NoveltyVerdict(**resp)
        )
        return state

    async def arun(
        self, *, description: str, explanation_summary: str
    ) -> NoveltyVerdict:
        state: NoveltyAdmissionState = {
            "description": description,
            "explanation_summary": explanation_summary,
        }
        final = await self.graph.ainvoke(state)
        return final["result"]
