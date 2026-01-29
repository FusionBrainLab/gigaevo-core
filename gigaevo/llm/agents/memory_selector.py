"""Memory selector agent for choosing relevant memory ideas for mutation."""

from typing import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter


class MemorySelectionOutput(BaseModel):
    """Structured output for memory selection."""

    ideas: list[str] = Field(
        default_factory=list,
        description="Top memory ideas relevant to the mutation prompt",
    )


class MemorySelectorState(TypedDict):
    """State for memory selector agent."""

    mutation_prompt: str
    memory_text: str
    max_ideas: int
    messages: list[BaseMessage]
    llm_response: MemorySelectionOutput
    ideas: list[str]


class MemorySelectorAgent(LangGraphAgent):
    """Agent that selects the most relevant memory ideas for a mutation."""

    StateSchema = MemorySelectorState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        llm = llm.with_structured_output(MemorySelectionOutput)
        super().__init__(llm)

    def build_prompt(self, state: MemorySelectorState) -> MemorySelectorState:
        user_prompt = self.user_prompt_template.format(
            mutation_prompt=state["mutation_prompt"],
            memory_text=state["memory_text"],
            max_ideas=state["max_ideas"],
        )
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        return state

    def parse_response(self, state: MemorySelectorState) -> MemorySelectorState:
        response: MemorySelectionOutput = state["llm_response"]
        max_ideas = max(0, int(state["max_ideas"]))
        ideas = [i.strip() for i in response.ideas if i and i.strip()]
        if max_ideas:
            ideas = ideas[:max_ideas]
        state["ideas"] = ideas
        return state

    async def arun(
        self, *, mutation_prompt: str, memory_text: str, max_ideas: int = 3
    ) -> list[str]:
        initial_state: MemorySelectorState = {
            "mutation_prompt": mutation_prompt,
            "memory_text": memory_text,
            "max_ideas": max_ideas,
            "messages": [],
            "llm_response": None,  # type: ignore
            "ideas": [],
        }
        final_state = await self.graph.ainvoke(initial_state)
        return final_state["ideas"]
