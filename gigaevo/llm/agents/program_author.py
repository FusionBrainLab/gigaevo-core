"""Author one holistic strategy hypothesis from a strong exemplar program."""

from __future__ import annotations

from typing import Any, Literal, Self, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, model_validator

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.agents.card_author import AuthoredCard
from gigaevo.llm.models import MultiModelRouter
from gigaevo.llm.schema_compat import portable_json_schema
from gigaevo.memory.write.decisions import WriteDecision


class ProgramAuthorResponse(BaseModel):
    decision: Literal[WriteDecision.DROP, WriteDecision.NEW]
    card: AuthoredCard | None

    @model_validator(mode="after")
    def _consistent_decision(self) -> Self:
        if self.decision is WriteDecision.DROP and self.card is not None:
            raise ValueError("DROP requires card=null")
        if self.decision is WriteDecision.NEW and self.card is None:
            raise ValueError("NEW requires one authored card")
        return self


class ProgramAuthorState(TypedDict, total=False):
    code: str
    fitness: float | None
    higher_is_better: bool
    archive_rank: int | None
    messages: list[BaseMessage]
    llm_response: Any
    result: ProgramAuthorResponse
    metadata: dict


class ProgramAuthorAgent(LangGraphAgent):
    StateSchema = ProgramAuthorState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        schema = portable_json_schema(ProgramAuthorResponse.model_json_schema())
        super().__init__(llm.with_structured_output(schema))

    def build_prompt(self, state: ProgramAuthorState) -> ProgramAuthorState:
        fitness = state.get("fitness")
        fitness_line = "(unknown)" if fitness is None else f"{fitness}"
        user = self.user_prompt_template.format(
            fitness=fitness_line,
            fitness_direction=(
                "higher is better" if state["higher_is_better"] else "lower is better"
            ),
            archive_rank=state.get("archive_rank") or "(unknown)",
            code=state["code"],
        )
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user),
        ]
        return state

    def parse_response(self, state: ProgramAuthorState) -> ProgramAuthorState:
        resp = state["llm_response"]
        state["result"] = (
            resp
            if isinstance(resp, ProgramAuthorResponse)
            else ProgramAuthorResponse(**resp)
        )
        return state

    async def arun(
        self,
        *,
        code: str,
        fitness: float | None,
        higher_is_better: bool,
        archive_rank: int | None = None,
    ) -> ProgramAuthorResponse:
        state: ProgramAuthorState = {
            "code": code,
            "fitness": fitness,
            "higher_is_better": higher_is_better,
            "archive_rank": archive_rank,
        }
        final = await self.graph.ainvoke(state)
        return final["result"]
