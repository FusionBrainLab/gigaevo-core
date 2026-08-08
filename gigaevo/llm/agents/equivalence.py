"""Kind-aware equivalence check for an authored card candidate."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.llm.schema_compat import portable_json_schema
from gigaevo.memory.cards import Card, CardKind, card_brief
from gigaevo.memory.write.decisions import WriteDecision


class AxisVerdict(StrEnum):
    """Semantic relation for one load-bearing program-strategy axis."""

    MATCH = "match"
    DIFFERENT = "different"
    UNSPECIFIED = "unspecified"


class ProgramAxisComparison(BaseModel):
    """Factorized comparison against the best-matching offered program family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    neighbor_id: str = Field(
        min_length=1,
        description="Exact offered program-card id being compared.",
    )
    applicability: AxisVerdict
    representation_or_state: AxisVerdict
    core_procedure: AxisVerdict
    decision_logic: AxisVerdict
    update_or_output_policy: AxisVerdict
    essential_constraints: AxisVerdict

    @property
    def equivalent(self) -> bool:
        return all(
            verdict is AxisVerdict.MATCH
            for verdict in (
                self.applicability,
                self.representation_or_state,
                self.core_procedure,
                self.decision_logic,
                self.update_or_output_policy,
                self.essential_constraints,
            )
        )


class EquivalenceResponse(BaseModel):
    comparison_summary: str = Field(
        min_length=1,
        description="Concise decisive match or mismatch in applicability and "
        "load-bearing action; never stored on the card.",
    )
    program_axes: ProgramAxisComparison | None = Field(
        description="Required for a program candidate and null for an insight. "
        "Compare the best-matching offered program neighbor axis by axis.",
    )
    decision: Literal[WriteDecision.NEW, WriteDecision.EQUIVALENT]
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
        llm: BaseChatModel | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        schema = portable_json_schema(EquivalenceResponse.model_json_schema())
        super().__init__(llm.with_structured_output(schema))

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
        result = (
            response
            if isinstance(response, EquivalenceResponse)
            else EquivalenceResponse(**response)
        )
        if state["candidate"].kind is CardKind.PROGRAM:
            axes = result.program_axes
            if axes is None:
                raise ValueError("program equivalence requires program_axes")
            offered_ids = {card.id for card in state["neighbors"]}
            if axes.neighbor_id not in offered_ids:
                raise ValueError(
                    f"program_axes neighbor {axes.neighbor_id!r} was not offered"
                )
            if (
                result.decision is WriteDecision.EQUIVALENT
                and result.target_id != axes.neighbor_id
            ):
                raise ValueError(
                    "EQUIVALENT target_id disagrees with program_axes neighbor_id"
                )
            equivalent = axes.equivalent
            result = result.model_copy(
                update={
                    "decision": (
                        WriteDecision.EQUIVALENT if equivalent else WriteDecision.NEW
                    ),
                    "target_id": axes.neighbor_id if equivalent else "",
                }
            )
        state["result"] = result
        return state

    async def arun(
        self, *, candidate: Card, neighbors: list[Card]
    ) -> EquivalenceResponse:
        if not neighbors:
            raise ValueError("equivalence requires at least one offered neighbor")
        final = await self.graph.ainvoke(
            {"candidate": candidate, "neighbors": neighbors}
        )
        return final["result"]
