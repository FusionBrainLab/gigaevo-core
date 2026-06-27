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

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.agents.reconcile import LibrarianCard
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.shared_memory.card_conversion import AnyCard


class ConsolidateState(TypedDict, total=False):
    card_a: AnyCard
    card_b: AnyCard
    messages: list[BaseMessage]
    llm_response: Any
    result: LibrarianCard
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
        super().__init__(llm.with_structured_output(LibrarianCard))

    def build_prompt(self, state: ConsolidateState) -> ConsolidateState:
        a, b = state["card_a"], state["card_b"]
        user = self.user_prompt_template.format(
            card_a=a.description,
            card_b=b.description,
        )
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user),
        ]
        return state

    def parse_response(self, state: ConsolidateState) -> ConsolidateState:
        resp = state["llm_response"]
        state["result"] = (
            resp if isinstance(resp, LibrarianCard) else LibrarianCard(**resp)
        )
        return state

    async def arun(self, *, card_a: AnyCard, card_b: AnyCard) -> LibrarianCard:
        state: ConsolidateState = {
            "card_a": card_a,
            "card_b": card_b,
        }
        final = await self.graph.ainvoke(state)
        return final["result"]
