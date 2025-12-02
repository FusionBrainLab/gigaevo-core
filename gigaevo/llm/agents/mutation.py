import re
from typing import Any, NotRequired, TypedDict

import diffpatch
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from gigaevo.evolution.mutation.context import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.program import Program


class MutationStructuredOutput(BaseModel):
    """Structured output from the mutation LLM.

    Simplified schema to reduce cognitive overhead and let LLM focus on code quality.
    """

    archetype: str = Field(
        description="Selected evolutionary archetype (e.g., 'Precision Optimization', 'Computational Reinvention')"
    )
    justification: str = Field(
        description="2-3 sentences: which insights acted on, strategy used, expected mechanism of improvement"
    )
    insights_used: list[str] = Field(
        default_factory=list,
        description="Flat list of insight strings that were acted on (verbatim from input)",
    )
    code: str = Field(description="The mutated Python program code")


# Metadata key for storing structured mutation output
MUTATION_OUTPUT_METADATA_KEY = "mutation_output"


class MutationPromptFields(BaseModel):
    """
    Example template:
        "Mutate {count} parent programs:\n{parent_blocks}"
    """

    count: int = Field(description="Number of parent programs")
    parent_blocks: str = Field(
        description="Formatted parent program blocks with code, metrics, insights"
    )


class MutationState(TypedDict):
    """State for mutation agent."""

    input: list[Program]
    mutation_mode: str
    messages: list[BaseMessage]
    llm_response: Any
    final_code: str
    mutation_label: str
    # Fields set during prompt building (optional initially)
    system_prompt: NotRequired[str]
    user_prompt: NotRequired[str]
    # Fields set during response parsing (optional initially)
    parsed_output: NotRequired[dict[str, Any]]
    structured_output: NotRequired[MutationStructuredOutput]
    error: NotRequired[str]


class MutationAgent(LangGraphAgent):
    """Agent for LLM-based code mutation.

    This agent handles the complete workflow of mutating programs:
    1. Build prompt from parent programs using pre-formatted mutation context
    2. Call LLM to generate structured output (archetype, justification, code)
    3. Extract and parse the structured output (handling diffs if needed)

    Attributes:
        mutation_mode: "rewrite" or "diff"
        system_prompt: System prompt
        user_prompt_template: User prompt template string
        structured_llm: LLM configured for structured output
    """

    StateSchema = MutationState

    def __init__(
        self,
        llm: ChatOpenAI | MultiModelRouter,
        system_prompt: str,
        user_prompt_template: str,
        mutation_mode: str = "rewrite",
    ):
        """Initialize mutation agent.

        Args:
            llm: LangChain chat model or router
            mutation_mode: "rewrite" or "diff"
            system_prompt: System prompt string
            user_prompt_template: User prompt template string
        """
        self.mutation_mode = mutation_mode
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

        # Create structured output LLM
        self.structured_llm = llm.with_structured_output(MutationStructuredOutput)

        super().__init__(llm)

    async def arun(self, input: list[Program], mutation_mode: str) -> dict:
        """Execute mutation agent.

        Args:
            input: List of parent programs to mutate
            mutation_mode: Mutation mode

        Returns:
            Dict with 'code', 'structured_output', and other mutation results
        """
        initial_state: MutationState = {
            "input": input,
            "mutation_mode": mutation_mode,
            "messages": [],
            "llm_response": None,
            "final_code": "",
            "mutation_label": "",
        }

        final_state = await self.graph.ainvoke(initial_state)
        return final_state.get("parsed_output", {})

    async def acall_llm(self, state: MutationState) -> MutationState:
        """Call LLM with structured output.

        Uses the structured LLM to get a MutationStructuredOutput response.

        Args:
            state: State with messages field

        Returns:
            Updated state with llm_response and structured_output fields
        """
        try:
            structured_response = await self.structured_llm.ainvoke(state["messages"])
            state["llm_response"] = structured_response
            state["structured_output"] = structured_response

            logger.debug(
                f"[MutationAgent] Received structured output with archetype: "
                f"{structured_response.archetype}"
            )

        except Exception as e:
            logger.error(f"[MutationAgent] Structured LLM call failed: {e}")
            state["error"] = str(e)
            state["llm_response"] = None

        return state

    def build_prompt(self, state: MutationState) -> MutationState:
        """Build mutation prompt from parent programs.

        Uses pre-formatted mutation context from MutationContextStage that includes:
        - Metrics (formatted)
        - Insights
        - Family tree lineage

        Args:
            state: Current state with parents field

        Returns:
            Updated state with messages field
        """
        parents = state["input"]

        # Build parent blocks - code + mutation context
        parent_blocks = []
        for i, p in enumerate(parents):
            # Get pre-formatted mutation context from metadata
            formatted_context = p.metadata.get(MUTATION_CONTEXT_METADATA_KEY)

            block = f"""=== Parent {i + 1} ===
```python
{p.code}
```

{formatted_context}
"""
            parent_blocks.append(block)

        # Use Pydantic model for type-safe formatting
        prompt_fields = MutationPromptFields(
            count=len(parents), parent_blocks="\n\n".join(parent_blocks)
        )

        user_prompt = self.user_prompt_template.format(**prompt_fields.model_dump())

        # Store prompts in state for logging
        state["system_prompt"] = self.system_prompt
        state["user_prompt"] = user_prompt

        # Build messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        state["messages"] = messages

        logger.debug(
            f"[MutationAgent] Built prompt with {len(parents)} parents "
            f"(system: {len(self.system_prompt)} chars, "
            f"user: {len(user_prompt)} chars)"
        )

        return state

    def parse_response(self, state: MutationState) -> MutationState:
        """Parse LLM structured response to extract code and metadata.

        Handles both rewrite mode (direct code from structured output) and diff mode
        (extract and apply diff from code field).

        Args:
            state: Current state with llm_response (structured output) field

        Returns:
            Updated state with parsed_output field containing final code and metadata
        """
        structured_output: MutationStructuredOutput | None = state.get(
            "structured_output"
        )

        if structured_output is None:
            error_msg = state.get("error", "No structured output received")
            logger.error(f"[MutationAgent] No structured output: {error_msg}")
            state["parsed_output"] = {
                "code": "",
                "structured_output": None,
                "error": error_msg,
            }
            return state

        try:
            # Get code from structured output
            code_from_llm = structured_output.code

            if state["mutation_mode"] == "diff":
                # Apply diff to parent code
                parents = state["input"]
                if len(parents) != 1:
                    raise ValueError("Diff mode requires exactly 1 parent")

                parent_code = parents[0].code
                # The code field contains the diff in diff mode
                final_code = self._apply_diff_and_extract(parent_code, code_from_llm)
            else:
                # In rewrite mode, clean up the code (remove any remaining fences)
                final_code = self._extract_code_block(code_from_llm)
                # If no code block markers found, use as-is
                if final_code == code_from_llm.strip():
                    final_code = code_from_llm.strip()

            state["final_code"] = final_code

            # Convert structured output to dict for storage
            structured_dict = structured_output.model_dump()

            state["parsed_output"] = {
                "code": final_code,
                "structured_output": structured_dict,
                "archetype": structured_output.archetype,
                "justification": structured_output.justification,
                "insights_used": structured_output.insights_used,
            }

            logger.debug(
                f"[MutationAgent] Extracted code ({len(final_code)} chars) "
                f"with archetype: {structured_output.archetype}"
            )

        except Exception as e:
            logger.error(f"[MutationAgent] Failed to parse structured response: {e}")
            state["error"] = str(e)
            state["parsed_output"] = {
                "code": "",
                "structured_output": (
                    structured_output.model_dump() if structured_output else None
                ),
                "error": str(e),
            }

        return state

    def _extract_code_block(self, text: str) -> str:
        """Extract outer fenced code block from LLM response.

        Treats only fences at start-of-line as valid markers to avoid
        premature closing on backticks inside code (e.g., docstrings).

        Args:
            text: LLM response text

        Returns:
            Extracted code string
        """
        # Find first opening fence at start-of-line
        open_match = re.search(r"(?m)^```(?:[a-zA-Z0-9_+\-]+)?\s*$", text)
        if not open_match:
            return text.strip()

        # Find closing fence after opener
        after_open = text[open_match.end() :]
        close_match = re.search(r"(?m)^```\s*$", after_open)
        if not close_match:
            return text.strip()

        code_block = after_open[: close_match.start()]

        # Trim single leading newline if present
        if code_block.startswith("\n"):
            code_block = code_block[1:]

        return code_block.rstrip()

    def _apply_diff_and_extract(self, original_code: str, response_text: str) -> str:
        """Extract diff from response and apply to original code.

        Args:
            original_code: Original parent code
            response_text: LLM response containing diff

        Returns:
            Patched code

        Raises:
            ValueError: If diff is empty or patch fails
        """
        diff_text = self._extract_code_block(response_text)
        if not diff_text.strip():
            raise ValueError("Empty diff returned by LLM")

        try:
            return diffpatch.apply_patch(original_code, diff_text)
        except Exception as e:
            raise ValueError(f"Failed to apply patch: {e}") from e
