from datetime import UTC, datetime
import os
import re
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

import diffpatch
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from gigaevo.evolution.mutation.context import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.programs.program import Program

if TYPE_CHECKING:
    from gigaevo.programs.metrics.context import MetricsContext
    from gigaevo.prompts.fetcher import PromptFetcher


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
    code: str = Field(
        description=(
            "The complete mutated Python source code. "
            "Must be valid Python (imports, function definitions, etc). "
            "NEVER put JSON or a response template here — only Python code."
        )
    )


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
    # Prompt tracking ID (None for fixed prompts, sha256[:16] for co-evolved prompts)
    prompt_id: NotRequired[str | None]
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
        # Optional: enable dynamic prompt fetching
        prompt_fetcher: "PromptFetcher | None" = None,
        task_description: str = "",
        metrics_context: "MetricsContext | None" = None,
    ):
        """Initialize mutation agent.

        Args:
            llm: LangChain chat model or router
            mutation_mode: "rewrite" or "diff"
            system_prompt: System prompt string (static or initial value)
            user_prompt_template: User prompt template string
            prompt_fetcher: Optional fetcher for dynamic prompt co-evolution.
                When set and is_dynamic=True, system_prompt is refreshed on
                every build_prompt() call. For FixedDirPromptFetcher, the
                static system_prompt is used without re-fetching.
            task_description: Task description for prompt template formatting
                (required when prompt_fetcher.is_dynamic is True)
            metrics_context: Metrics context for prompt template formatting
                (required when prompt_fetcher.is_dynamic is True)
        """
        self.mutation_mode = mutation_mode
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

        # Dynamic prompt fetching support
        self._prompt_fetcher = prompt_fetcher
        self._task_description = task_description
        if metrics_context is not None:
            from gigaevo.programs.metrics.formatter import MetricsFormatter

            self._metrics_formatter: MetricsFormatter | None = MetricsFormatter(
                metrics_context
            )
        else:
            self._metrics_formatter = None

        # Create structured output LLM
        self.structured_llm = llm.with_structured_output(MutationStructuredOutput)

        super().__init__(llm)

    _PROMPT_LOG_DIR = os.environ.get("GIGAEVO_PROMPT_LOG_DIR", "")

    def _dump_prompt_to_file(
        self, prompt_id: str | None, system: str, user: str
    ) -> None:
        """Write full system+user prompts to a log file for offline inspection."""
        log_dir = self._PROMPT_LOG_DIR
        if not log_dir:
            return
        try:
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            pid = prompt_id or "fixed"
            path = os.path.join(log_dir, f"{ts}_{pid[:12]}.txt")
            with open(path, "w") as f:
                f.write(f"=== PROMPT DUMP {ts} ===\n")
                f.write(f"prompt_id: {prompt_id}\n\n")
                f.write("=== SYSTEM PROMPT ===\n")
                f.write(system)
                f.write("\n\n=== USER PROMPT ===\n")
                f.write(user)
                f.write("\n")
        except Exception as exc:
            logger.debug(f"[MutationAgent] prompt dump failed: {exc}")

    async def arun(self, input: list[Program], mutation_mode: str) -> dict:
        """Execute mutation agent.

        Args:
            input: List of parent programs to mutate
            mutation_mode: Mutation mode

        Returns:
            Dict with 'code', 'structured_output', 'prompt_id', and other results
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
        result = final_state.get("parsed_output", {})
        # Forward prompt_id from state into result for operator to stamp in metadata
        result["prompt_id"] = final_state.get("prompt_id")
        return result

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

            # Log model used (if LLM is a router)
            model_used = None
            if isinstance(self.llm, MultiModelRouter):
                model_used = self.llm.get_last_model()
            logger.debug(
                "[MutationAgent] Received structured output — archetype: {}, model: {}",
                structured_response.archetype,
                model_used or "(single model)",
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

        If a dynamic prompt_fetcher is configured (is_dynamic=True), refreshes the
        system prompt from the co-evolving archive and stamps prompt_id in state.

        Args:
            state: Current state with parents field

        Returns:
            Updated state with messages field and optional prompt_id
        """
        # Refresh system and user prompts from dynamic fetcher if available
        if (
            self._prompt_fetcher is not None
            and self._prompt_fetcher.is_dynamic
            and self._metrics_formatter is not None
        ):
            fetched_sys = self._prompt_fetcher.fetch("mutation", "system")
            self.system_prompt = fetched_sys.text.format(
                task_description=self._task_description,
                metrics_description=self._metrics_formatter.format_metrics_description(),
            )
            state["prompt_id"] = fetched_sys.prompt_id
            # Also refresh user prompt template if a co-evolved version is available
            fetched_user = self._prompt_fetcher.fetch("mutation", "user")
            if fetched_user.prompt_id is not None:
                self.user_prompt_template = fetched_user.text
        else:
            state["prompt_id"] = None

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

        logger.info(
            f"[MutationAgent] Built prompt with {len(parents)} parents "
            f"(system: {len(self.system_prompt)} chars, "
            f"user: {len(user_prompt)} chars, "
            f"prompt_id={state.get('prompt_id', 'N/A')})"
        )
        # Dump full prompts to file for offline verification
        self._dump_prompt_to_file(
            state.get("prompt_id"), self.system_prompt, user_prompt
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

            # Guard: reject code that is a JSON template echoed back instead of Python
            if "def " not in final_code and final_code.lstrip().startswith("{"):
                raise ValueError(
                    "LLM returned JSON template as code instead of Python. "
                    f"Code starts with: {final_code[:80]!r}"
                )

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
