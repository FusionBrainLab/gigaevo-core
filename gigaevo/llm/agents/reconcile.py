"""Reconcile agent: a single LangGraph LLM hop on the memory write path.

Given the parent->child diff (ground truth) and the nearest existing cards, the
librarian decides for each generalizable lever whether it is NEW, a DUPLICATE of
an existing card, or a MERGE into one — and authors clean, transferable card
prose for it. Empty ``items`` means the diff carried no generalizable lever
(drop). Structured output is bound at construction via ``with_structured_output``
so the router negotiates the method (do not force ``function_calling``).

Prompts follow the insights/lineage convention: the system prompt (with the
task baked into its CONTEXT) and the user template are injected at construction
via :func:`gigaevo.llm.agents.factories.create_reconcile_agent`.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.cards import Card, card_brief


class LibrarianCard(BaseModel):
    description: str = Field(
        description="Clean, generalizable mechanism framed as the failure mode "
        "it bypasses; not a tautology or bare factoid."
    )
    explanation_summary: str = Field(
        default="",
        description="One sentence condensing WHY the lever works — the causal "
        "reason it escapes the failure mode, not a restatement of the description. "
        "Indexed as its own retrieval channel, so always author it.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Semantic retrieval tags; plain words, no machine prefixes.",
    )


class ReconcileItem(BaseModel):
    decision: Literal["NEW", "DUPLICATE", "MERGE"] = Field(
        description="NEW: novel lever. DUPLICATE: same lever as target_id "
        "(drop, bump provenance). MERGE: combine with target_id into the "
        "authored union card."
    )
    card: LibrarianCard = Field(description="The authored card for this lever.")
    target_id: str = Field(
        default="",
        description="Existing neighbor id for DUPLICATE/MERGE; empty for NEW.",
    )


class ReconcileResponse(BaseModel):
    items: list[ReconcileItem] = Field(
        default_factory=list,
        description="0..N levers extracted from the diff; empty = no "
        "generalizable lever (drop).",
    )


class ReconcileState(TypedDict, total=False):
    base_parent_code: str
    child_code: str
    note: str
    neighbors: list[Card]
    messages: list[BaseMessage]
    llm_response: Any
    result: ReconcileResponse
    metadata: dict


class ReconcileAgent(LangGraphAgent):
    StateSchema = ReconcileState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        super().__init__(llm.with_structured_output(ReconcileResponse))

    def build_prompt(self, state: ReconcileState) -> ReconcileState:
        neighbors = "\n".join(
            f"- {c.id}: {card_brief(c)}" for c in state.get("neighbors", [])
        )
        user = self.user_prompt_template.format(
            base_parent_code=state["base_parent_code"],
            child_code=state["child_code"],
            note=state["note"],
            neighbors=neighbors or "(none)",
        )
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user),
        ]
        return state

    def parse_response(self, state: ReconcileState) -> ReconcileState:
        resp = state["llm_response"]
        state["result"] = (
            resp if isinstance(resp, ReconcileResponse) else ReconcileResponse(**resp)
        )
        return state

    async def arun(
        self,
        *,
        base_parent_code: str,
        child_code: str,
        note: str,
        neighbors: list[Card],
    ) -> ReconcileResponse:
        state: ReconcileState = {
            "base_parent_code": base_parent_code,
            "child_code": child_code,
            "note": note,
            "neighbors": neighbors,
        }
        final = await self.graph.ainvoke(state)
        return final["result"]
