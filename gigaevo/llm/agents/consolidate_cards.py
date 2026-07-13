"""Consolidate agent: synthesize one union card from two drifted near-dups.

The periodic consolidation pass finds two existing cards that name the same
generalizable lever and asks this agent to author a single canonical card whose
prose covers both — the union framed as the failure mode it bypasses, not a
concatenation. Returns the same ``LibrarianCard`` schema the reconcile hop uses,
so the survivor card is overwritten with clean, transferable prose.

Prompts follow the insights/lineage convention: the system prompt (with the
task baked into its CONTEXT) and the user template are injected at construction
via :func:`gigaevo.llm.agents.factories.create_consolidate_agent`.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.agents.reconcile import LibrarianCard, arbiter_card_brief
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.cards import Card


class ConsolidateDecision(BaseModel):
    """The agent's merge-or-abstain ruling on two nearest-neighbor cards.

    The consolidation pass surfaces NEAR cards as merge *candidates*; this agent
    is the precision arbiter. ``merge=False`` keeps both cards (they only drifted
    close, they are not the same lever) so a generous candidate gate can never
    force-merge distinct levers.
    """

    merge: bool = Field(
        description="True iff Card A and Card B name the SAME generalizable lever "
        "and must be folded into one canonical card; False to keep them as two "
        "distinct cards."
    )
    card: LibrarianCard | None = Field(
        default=None,
        description="The authored union card — populated only when merge is True; "
        "null when abstaining.",
    )


class ConsolidateState(TypedDict, total=False):
    card_a: Card
    card_b: Card
    messages: list[BaseMessage]
    llm_response: Any
    result: ConsolidateDecision
    metadata: dict


class ConsolidateAgent(LangGraphAgent):
    StateSchema = ConsolidateState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        super().__init__(llm.with_structured_output(ConsolidateDecision))

    def build_prompt(self, state: ConsolidateState) -> ConsolidateState:
        a, b = state["card_a"], state["card_b"]
        user = self.user_prompt_template.format(
            card_a=arbiter_card_brief(a),
            card_b=arbiter_card_brief(b),
        )
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user),
        ]
        return state

    def parse_response(self, state: ConsolidateState) -> ConsolidateState:
        resp = state["llm_response"]
        state["result"] = (
            resp
            if isinstance(resp, ConsolidateDecision)
            else ConsolidateDecision(**resp)
        )
        return state

    async def arun(self, *, card_a: Card, card_b: Card) -> ConsolidateDecision:
        state: ConsolidateState = {
            "card_a": card_a,
            "card_b": card_b,
        }
        final = await self.graph.ainvoke(state)
        return final["result"]
