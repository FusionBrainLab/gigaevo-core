"""Base class for LangGraph agents.

This module provides the abstract LangGraphAgent base class that all
LLM-based agents inherit from. Each agent defines its own domain-specific
state schema using TypedDict.
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from src.llm.models import MultiModelRouter


class LangGraphAgent(ABC):
    """Abstract base for all LLM agents using LangGraph.

    Each agent must:
    1. Define its own StateSchema (TypedDict) with domain-specific fields
    2. Implement build_prompt() to create LangChain messages
    3. Implement parse_response() to extract structured output

    The base class provides:
    - Generic async LLM invocation (acall_llm)
    - Graph construction (build_prompt → call_llm → parse_response)
    - Agent execution (arun method)

    Example:
        >>> class MyState(TypedDict):
        ...     data: str
        ...     messages: list[BaseMessage]
        ...     llm_response: AIMessage
        ...     result: str
        >>>
        >>> class MyAgent(LangGraphAgent):
        ...     StateSchema = MyState
        ...
        ...     def build_prompt(self, state):
        ...         state["messages"] = [HumanMessage(state["data"])]
        ...         return state
        ...
        ...     def parse_response(self, state):
        ...         state["result"] = state["llm_response"].content
        ...         return state
    """

    # Subclasses must define their StateSchema
    StateSchema: type

    def __init__(self, llm: ChatOpenAI | MultiModelRouter):
        """Initialize agent with LLM.

        Args:
            llm: LangChain chat model or multi-model router
        """
        self.llm = llm
        self.graph = self._build_graph()
        logger.info(f"[{self.__class__.__name__}] Initialized")

    @abstractmethod
    def build_prompt(self, state: Any) -> Any:
        """Build LangChain messages from domain state.

        Must populate state["messages"] with appropriate message list.

        Args:
            state: Domain-specific state dict

        Returns:
            Updated state with messages populated
        """
        pass

    async def acall_llm(self, state: Any) -> Any:
        """Generic async LLM call.

        Invokes the LLM with messages from state and stores response.
        Also tracks which model was used for debugging.

        Args:
            state: State with messages field

        Returns:
            Updated state with llm_response field
        """
        response = await self.llm.ainvoke(state["messages"])
        state["llm_response"] = response

        # Track metadata
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["model_used"] = "llm"

        return state

    @abstractmethod
    def parse_response(self, state: Any) -> Any:
        """Parse LLM response into domain output.

        Extracts relevant information from state["llm_response"]
        and stores it in a domain-specific output field.

        Args:
            state: State with llm_response field

        Returns:
            Updated state with parsed output
        """
        pass

    def _build_graph(self) -> CompiledStateGraph:
        """Build LangGraph execution graph.

        Creates a simple 3-node linear graph:
        build_prompt → call_llm → parse_response → END

        Returns:
            Compiled LangGraph
        """
        workflow = StateGraph(self.StateSchema)

        workflow.add_node("build_prompt", self.build_prompt)
        workflow.add_node("call_llm", self.acall_llm)
        workflow.add_node("parse_response", self.parse_response)

        workflow.set_entry_point("build_prompt")
        workflow.add_edge("build_prompt", "call_llm")
        workflow.add_edge("call_llm", "parse_response")
        workflow.add_edge("parse_response", END)

        return workflow.compile()

    @abstractmethod
    async def arun(self, *args, **kwargs) -> Any:
        """Execute agent and return result.

        Each agent defines its own signature based on domain needs.
        For example:
        - InsightsAgent.arun(program: Program) -> list[dict]
        - LineageAgent.arun(parent: Program, child: Program) -> list[dict]

        This method should:
        1. Create initial state with domain-specific fields
        2. Invoke the graph
        3. Return the parsed output
        """
        pass
