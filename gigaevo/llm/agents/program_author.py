"""Program-author agent: exemplar card prose from a top-fitness program.

Authors one card describing what a high-fitness exemplar program does and *why*
it scores well — a transferable mechanism, not a line-by-line trace. Used by the
librarian to fill ``ProgramCard.description`` so no exemplar card carries a
borrowed or empty (``pending_analysis``) description.

Prompts follow the insights/lineage convention: the system prompt (with the
task baked into its CONTEXT) and the user template are injected at construction
via :func:`gigaevo.llm.agents.factories.create_program_author_agent`.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter


class ProgramAuthorResponse(BaseModel):
    description: str = Field(
        description="What the exemplar does and why it scores well; transferable "
        "mechanism, not a line-by-line trace."
    )
    explanation_summary: str = Field(
        default="",
        description="One sentence condensing WHY the exemplar scores well — the "
        "causal reason, not a restatement of the description. Indexed as its own "
        "retrieval channel, so always author it.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Semantic retrieval tags; plain words, no machine prefixes.",
    )


class ProgramAuthorState(TypedDict, total=False):
    code: str
    fitness: float | None
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
        super().__init__(llm.with_structured_output(ProgramAuthorResponse))

    def build_prompt(self, state: ProgramAuthorState) -> ProgramAuthorState:
        fitness = state.get("fitness")
        fitness_line = "(unknown)" if fitness is None else f"{fitness}"
        user = self.user_prompt_template.format(
            fitness=fitness_line,
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

    async def arun(self, *, code: str, fitness: float | None) -> ProgramAuthorResponse:
        state: ProgramAuthorState = {
            "code": code,
            "fitness": fitness,
        }
        final = await self.graph.ainvoke(state)
        return final["result"]
