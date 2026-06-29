"""Task-summary agent: a one-line condensation of the full task description.

The librarian stamps every card with both the full ``task_description`` and a
short ``task_description_summary``. This agent produces that summary once per run
from the run's task text, so the summary is a genuine condensation rather than
the full task text duplicated verbatim.

Prompts follow the insights/lineage convention: the (task-agnostic) system
prompt and the ``{task_description}`` user template are injected at construction
via :func:`gigaevo.llm.agents.factories.create_task_summary_agent`.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter


class TaskSummaryResponse(BaseModel):
    summary: str = Field(
        description="One-line condensation of the task objective.",
    )


class TaskSummaryState(TypedDict, total=False):
    task_description: str
    messages: list[BaseMessage]
    llm_response: Any
    result: TaskSummaryResponse
    metadata: dict


class TaskSummaryAgent(LangGraphAgent):
    StateSchema = TaskSummaryState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        super().__init__(llm.with_structured_output(TaskSummaryResponse))

    def build_prompt(self, state: TaskSummaryState) -> TaskSummaryState:
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=self.user_prompt_template.format(
                    task_description=state["task_description"]
                )
            ),
        ]
        return state

    def parse_response(self, state: TaskSummaryState) -> TaskSummaryState:
        resp = state["llm_response"]
        state["result"] = (
            resp
            if isinstance(resp, TaskSummaryResponse)
            else TaskSummaryResponse(**resp)
        )
        return state

    async def arun(self, *, task_description: str) -> TaskSummaryResponse:
        state: TaskSummaryState = {"task_description": task_description}
        final = await self.graph.ainvoke(state)
        return final["result"]
