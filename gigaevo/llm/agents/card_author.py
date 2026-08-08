"""Author at most one transferable hypothesis from one mutation outcome."""

from __future__ import annotations

import difflib
from typing import Any, Literal, Self, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.llm.schema_compat import portable_json_schema
from gigaevo.memory.write.decisions import ArchiveStatus, WriteDecision

_MAX_DIFF_CHARS = 16_000


class AuthoredCard(BaseModel):
    """One actionable, conditional memory hypothesis."""

    description: str = Field(
        min_length=1,
        description="A conditional hypothesis naming one observable applicability "
        "condition, one implementable intervention or strategy, and its mechanism.",
    )
    explanation_summary: str = Field(
        min_length=1,
        description="The concise mechanism that makes the proposed action useful "
        "under its stated applicability condition.",
    )


class CardAuthorResponse(BaseModel):
    """A mutation produces either no durable idea or one candidate."""

    decision: Literal[WriteDecision.DROP, WriteDecision.NEW]
    card: AuthoredCard | None

    @model_validator(mode="after")
    def _consistent_decision(self) -> Self:
        if self.decision is WriteDecision.DROP and self.card is not None:
            raise ValueError("DROP requires card=null")
        if self.decision is WriteDecision.NEW and self.card is None:
            raise ValueError("NEW requires one authored card")
        return self


class CardAuthorState(TypedDict, total=False):
    base_parent_code: str
    child_code: str
    unified_diff: str
    mutation_report: str
    parent_fitness: float | None
    child_fitness: float
    signed_gain: float | None
    higher_is_better: bool
    archive_status: ArchiveStatus
    messages: list[BaseMessage]
    llm_response: Any
    result: CardAuthorResponse
    metadata: dict


class CardAuthorAgent(LangGraphAgent):
    StateSchema = CardAuthorState

    def __init__(
        self,
        llm: BaseChatModel | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
        fitness_key: str,
    ) -> None:
        if not fitness_key.strip():
            raise ValueError("fitness_key cannot be empty")
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.fitness_key = fitness_key.strip()
        schema = portable_json_schema(CardAuthorResponse.model_json_schema())
        super().__init__(llm.with_structured_output(schema))

    def build_prompt(self, state: CardAuthorState) -> CardAuthorState:
        user = self.user_prompt_template.format(
            fitness_key=self.fitness_key,
            base_parent_code=state["base_parent_code"],
            child_code=state["child_code"],
            unified_diff=state.get("unified_diff")
            or unified_diff(state["base_parent_code"], state["child_code"]),
            mutation_report=state["mutation_report"],
            parent_fitness=_render_metric(state.get("parent_fitness")),
            child_fitness=_render_metric(state["child_fitness"]),
            signed_gain=_render_metric(state.get("signed_gain")),
            fitness_direction=(
                "higher is better" if state["higher_is_better"] else "lower is better"
            ),
            archive_status=state["archive_status"].value,
        )
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user),
        ]
        return state

    def parse_response(self, state: CardAuthorState) -> CardAuthorState:
        response = state["llm_response"]
        state["result"] = (
            response
            if isinstance(response, CardAuthorResponse)
            else CardAuthorResponse(**response)
        )
        return state

    async def arun(
        self,
        *,
        base_parent_code: str,
        child_code: str,
        mutation_report: str,
        parent_fitness: float | None,
        child_fitness: float,
        signed_gain: float | None,
        higher_is_better: bool,
        archive_status: ArchiveStatus,
    ) -> CardAuthorResponse:
        state: CardAuthorState = {
            "base_parent_code": base_parent_code,
            "child_code": child_code,
            "unified_diff": unified_diff(base_parent_code, child_code),
            "mutation_report": mutation_report,
            "parent_fitness": parent_fitness,
            "child_fitness": child_fitness,
            "signed_gain": signed_gain,
            "higher_is_better": higher_is_better,
            "archive_status": archive_status,
        }
        final = await self.graph.ainvoke(state)
        return final["result"]


def unified_diff(base_parent_code: str, child_code: str) -> str:
    """Return a bounded parent-to-child diff for grounded hypothesis authoring."""
    lines = list(
        difflib.unified_diff(
            base_parent_code.splitlines(),
            child_code.splitlines(),
            fromfile="base_parent.py",
            tofile="child.py",
            lineterm="",
            n=3,
        )
    )
    if not lines:
        return "(No code differences detected)"
    diff = "\n".join(lines)
    if len(diff) <= _MAX_DIFF_CHARS:
        return diff
    return diff[: _MAX_DIFF_CHARS - 40] + "\n...[diff truncated]"


def _render_metric(value: float | None) -> str:
    return "(unknown)" if value is None else str(value)
