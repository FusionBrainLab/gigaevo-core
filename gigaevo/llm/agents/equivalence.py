"""Strict interventional equivalence check for an authored card candidate."""

from __future__ import annotations

from typing import Any, Self, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.cards import Card, card_brief
from gigaevo.memory.write.decisions import WriteDecision


class EquivalenceResponse(BaseModel):
    decision: WriteDecision
    target_id: str = Field(
        default="",
        description="Offered neighbor id for EQUIVALENT; empty for NEW.",
    )

    @model_validator(mode="after")
    def _consistent_decision(self) -> Self:
        if self.decision is WriteDecision.EQUIVALENT and not self.target_id.strip():
            raise ValueError("EQUIVALENT requires target_id")
        if self.decision is WriteDecision.NEW and self.target_id:
            raise ValueError("NEW requires an empty target_id")
        if self.decision is WriteDecision.DROP:
            raise ValueError("equivalence checking cannot drop an authored candidate")
        return self


class EquivalenceState(TypedDict, total=False):
    candidate: Card
    neighbors: list[Card]
    messages: list[BaseMessage]
    llm_response: Any
    result: EquivalenceResponse
    metadata: dict


class EquivalenceAgent(LangGraphAgent):
    StateSchema = EquivalenceState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        super().__init__(llm.with_structured_output(EquivalenceResponse))

    def build_prompt(self, state: EquivalenceState) -> EquivalenceState:
        candidate = state["candidate"]
        neighbors = "\n".join(
            f"- {card.id}: {card_brief(card)}" for card in state["neighbors"]
        )
        user = self.user_prompt_template.format(
            kind=candidate.kind.value,
            candidate=card_brief(candidate),
            neighbors=neighbors or "(none)",
        )
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user),
        ]
        return state

    def parse_response(self, state: EquivalenceState) -> EquivalenceState:
        response = state["llm_response"]
        state["result"] = (
            response
            if isinstance(response, EquivalenceResponse)
            else EquivalenceResponse(**response)
        )
        return state

    async def arun(
        self, *, candidate: Card, neighbors: list[Card]
    ) -> EquivalenceResponse:
        final = await self.graph.ainvoke(
            {"candidate": candidate, "neighbors": neighbors}
        )
        return final["result"]
