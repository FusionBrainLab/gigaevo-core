"""Scoring agent for program trait evaluation using LangGraph.

This agent assigns numeric scores to programs based on specific traits.
ALL LLM-related logic lives here - stages are just thin wrappers.
"""

from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from src.llm.agents.base import LangGraphAgent
from src.llm.models import MultiModelRouter
from src.programs.program import Program


class ProgramScore(BaseModel):
    """Structured score output."""

    score: float = Field(description="Numeric score for the program")


class ScoringState(TypedDict):
    """Complete state for scoring.

    This is the LangGraph state - it flows through all nodes.
    """

    # Input
    program: Program
    trait_description: str
    max_score: float

    # LLM interaction
    messages: list[BaseMessage]
    llm_response: AIMessage | ProgramScore

    # Output
    score: float

    # Metadata
    metadata: dict


class ScoringAgent(LangGraphAgent):
    StateSchema = ScoringState
    """Agent for scoring programs on specific traits.
    
    This agent does ALL the heavy lifting:
    - Formats prompts
    - Calls LLM
    - Parses structured output
    - Clips score to max
    
    Stages just call agent.arun(program, trait, max_score) and store results.
    """

    StateSchema = ScoringState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
        trait_description: str,
        max_score: float,
    ):
        """Initialize scoring agent.

        Args:
            llm: LangChain chat model or router
            system_prompt: System prompt (no template vars)
            user_prompt_template: User prompt template (with {code}, {trait_description}, {max_score})
            trait_description: Description of trait to score
            max_score: Maximum allowed score
        """
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.trait_description = trait_description
        self.max_score = max_score
        llm = llm.with_structured_output(ProgramScore)
        super().__init__(llm)

    def build_prompt(self, state: ScoringState) -> ScoringState:
        """Build scoring prompt - ALL formatting logic here."""
        program = state["program"]
        trait_description = state["trait_description"]
        max_score = state["max_score"]

        # Format user prompt (agent responsibility!)
        user_prompt = self.user_prompt_template.format(
            code=program.code,
            trait_description=trait_description,
            max_score=max_score,
        )

        # Create messages
        state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        return state

    def parse_response(self, state: ScoringState) -> ScoringState:
        """Parse LLM response (already validated by LangChain structured output)."""
        llm_response = state["llm_response"]

        # LangChain with_structured_output returns ProgramScore directly
        if not isinstance(llm_response, ProgramScore):
            raise ValueError(f"Expected ProgramScore, got {type(llm_response)}")

        # Clip score to max (agent responsibility!)
        max_score = state["max_score"]
        score = min(llm_response.score, max_score)

        state["score"] = score
        return state

    async def arun(self, program: Program) -> float:
        """Run scoring on a program.

        Args:
            program: Program to score

        Returns:
            ProgramScore
        """
        initial_state: ScoringState = {
            "program": program,  # Direct fields - no nesting!
            "trait_description": self.trait_description,
            "max_score": self.max_score,
            "messages": [],
            "llm_response": None,  # type: ignore
            "score": 0.0,
            "metadata": {"program_id": program.id},
        }

        final_state = await self.graph.ainvoke(initial_state)
        return final_state["score"]
