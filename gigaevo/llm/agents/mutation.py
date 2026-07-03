import ast
from datetime import UTC, datetime
import json
import os
import re
import time
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

import diffpatch
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.evolution.mutation.constants import (
    MUTATION_CONTEXT_METADATA_KEY,
    ArchetypeName,
)
from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.llm.models import (
    MultiModelRouter,
    get_last_token_usage,
    get_selected_model,
)
from gigaevo.llm.token_tracking import llm_stage_context
from gigaevo.monitoring.emit import emit as _emit_event
from gigaevo.monitoring.events import LLMCall
from gigaevo.programs.program import Program

if TYPE_CHECKING:
    from gigaevo.programs.metrics.context import MetricsContext
    from gigaevo.prompts.fetcher import PromptFetcher


class MutationChange(BaseModel):
    """Tracker-friendly description of one introduced change."""

    description: str = Field(
        description=(
            "Generalizable description of the introduced change, optionally followed "
            "by concrete specifics when they matter. Prefer `general pattern + "
            "concrete instance` over a narrow one-off description."
        )
    )
    explanation: str = Field(
        description=(
            "Explain why this change was introduced, why it helped for this "
            "program, and when possible why the same idea could transfer to future "
            "mutations."
        )
    )


class InsightCitation(BaseModel):
    """One cited entry from a parent's numbered '## Program Insights' list."""

    parent: int = Field(
        default=1,
        description=(
            "1-based parent number (as labelled '=== Parent N ===') whose "
            "insight list this refers to. Always 1 in single-parent mode."
        ),
    )
    insight: int = Field(
        description=(
            "The `N.` number of the insight in that parent's "
            "'## Program Insights' list."
        )
    )


def compute_citation_integrity(
    cited_pairs: list[tuple[str, int]],
    card_ids_used: list[str],
    messages: list[BaseMessage],
) -> dict[str, int]:
    """Count cited insight and card references that were offered in the prompt.

    Each parent block carries its own '## Program Insights' list numbered from
    1, so an insight citation is a (parent-label, insight-number) pair:
    grounded when the insight's rendered list marker ('\\n<N>. **[' — the
    numbering emitted by InsightsMutationContext.format) appears inside that
    parent's block. Card ids match exactly anywhere.

    Parent-label-agnostic: matches both '=== Parent N ===' (standard, numeric)
    and '=== Parent A [evaluation context] ===' (structured diff, letter). When
    a letter labels both a listing block and an evaluation-context block, the
    later (evaluation-context) block wins — that is where insights render.

    Purely observational (the counts land in child metadata for run statistics,
    nothing is gated on them — the hard credit gate is the base_selected ∩ used
    intersection at the write path).
    """
    rendered = "\n".join(str(m.content) for m in messages)
    parts = re.split(r"=== Parent (\w+)(?: evaluation context)? ===", rendered)
    blocks = {label: block for label, block in zip(parts[1::2], parts[2::2])}
    cited = [(label, n) for label, n in cited_pairs if label and n > 0]
    cards = [c.strip() for c in card_ids_used if c.strip()]
    return {
        "cited": len(cited),
        "grounded": sum(
            1 for label, n in cited if f"\n{n}. **[" in blocks.get(label, "")
        ),
        "cards_cited": len(cards),
        "cards_grounded": sum(1 for c in cards if c in rendered),
    }


class MutationStructuredOutput(BaseModel):
    """Structured output from the mutation LLM.

    Simplified schema to reduce cognitive overhead and let LLM focus on code quality.
    """

    archetype: ArchetypeName = Field(
        description="Selected evolutionary archetype (one of the 8 canonical names in ARCHETYPE_NAMES)."
    )
    justification: str = Field(
        description="2-3 sentences: which insights acted on, strategy used, expected mechanism of improvement"
    )
    insights_used: list[str] = Field(
        default_factory=list,
        description="Flat list of insight strings that were acted on (verbatim from input)",
    )
    insight_ids_used: list[InsightCitation] = Field(
        default_factory=list,
        description=(
            "Program Insights entries acted on, each cited as the parent number "
            "plus the `N.` number in that parent's numbered insight list. Each "
            "parent's list numbers from 1, so the parent number disambiguates. "
            "Empty when the lists are empty or none were used. Never invent "
            "numbers."
        ),
    )
    base_parent: int = Field(
        default=1,
        description=(
            "1-based number of the parent (as labelled '=== Parent N ===' above) "
            "whose overall structure this child keeps. If you blend both, pick the "
            "one it most resembles. Reward and context anchor to this parent."
        ),
    )
    card_ids_used: list[str] = Field(
        default_factory=list,
        description=(
            "Exact ids of the memory cards you applied. When you act on an insight "
            "carrying a `card: <id>` attribution, copy that id here verbatim — this "
            "is the only signal that credits the card. Never invent ids; leave empty "
            "if you applied no card-sourced insight."
        ),
    )
    changes: list[MutationChange] = Field(
        default_factory=list,
        description=(
            "Key introduced changes. Each item must contain a reusable or "
            "generalizable description plus an explanation of why the change was "
            "introduced."
        ),
    )
    code: str = Field(
        description=(
            "The complete mutated Python source code. "
            "Must be valid Python starting with imports or def statements. "
            "NEVER put JSON, format examples, or templates here. "
            "Use actual newlines between lines, not literal backslash-n."
        )
    )


# Re-export from canonical location for backward compatibility
MUTATION_OUTPUT_METADATA_KEY = MutationSpec.META_OUTPUT


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
    metadata: NotRequired[dict[str, Any]]
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
        Emits exactly one LLM_CALL canonical event per invocation (success or
        failure) so mutation LLM latency joins the same observability stream
        as LineageAgent / InsightsAgent — see ``gigaevo.monitoring.events``.

        Args:
            state: State with messages field

        Returns:
            Updated state with llm_response and structured_output fields
        """
        t0 = time.monotonic()
        error_type: str | None = None
        ok = False
        structured_response: Any = None
        try:
            with llm_stage_context(self.__class__.__name__):
                structured_response = await self.structured_llm.ainvoke(
                    state["messages"]
                )
            state["llm_response"] = structured_response
            state["structured_output"] = structured_response
            if "metadata" not in state:
                state["metadata"] = {}
            model_used = get_selected_model()
            if model_used:
                state["metadata"]["model_used"] = model_used
            ok = True

            logger.debug(
                "[MutationAgent] Received structured output — archetype: {}, model: {}",
                structured_response.archetype,
                model_used or "(single model)",
            )

        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"[MutationAgent] Structured LLM call failed: {e}")
            state["error"] = str(e)
            state["llm_response"] = None
        finally:
            try:
                model = getattr(self.llm, "model_name", None) or (
                    get_selected_model() or "unknown"
                )
                usage = get_last_token_usage()
                _emit_event(
                    LLMCall(
                        stage="MutationAgent",
                        endpoint="",
                        model=str(model),
                        attempt=1,
                        ok=ok,
                        latency_ms=(time.monotonic() - t0) * 1000.0,
                        tokens_in=usage.context if usage else 0,
                        tokens_out=usage.generated if usage else 0,
                        error_type=error_type,
                    )
                )
            except Exception:  # pragma: no cover — never fail the call on logging
                logger.opt(exception=True).debug(
                    "[MutationAgent] LLM_CALL emission failed"
                )

        return state

    def _refresh_prompts_from_fetcher(self, state: MutationState) -> None:
        """Refresh system and user prompts from the dynamic co-evolving fetcher.

        Stamps prompt_id in state for downstream tracking.
        Called only when prompt_fetcher.is_dynamic is True.
        """
        assert self._prompt_fetcher is not None
        assert self._metrics_formatter is not None
        fetched_sys = self._prompt_fetcher.fetch("mutation", "system")
        self.system_prompt = fetched_sys.text.format(
            task_description=self._task_description,
            metrics_description=self._metrics_formatter.format_metrics_description(),
        )
        state["prompt_id"] = fetched_sys.prompt_id
        fetched_user = self._prompt_fetcher.fetch("mutation", "user")
        if fetched_user.prompt_id is not None:
            self.user_prompt_template = fetched_user.text

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
        if (
            self._prompt_fetcher is not None
            and self._prompt_fetcher.is_dynamic
            and self._metrics_formatter is not None
        ):
            self._refresh_prompts_from_fetcher(state)
        else:
            state["prompt_id"] = None

        parents = state["input"]
        user_prompt = self.build_user_prompt(parents)

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

    def build_user_prompt(self, parents: list[Program]) -> str:
        """Build the mutation user prompt for a set of parents."""
        parent_blocks = self._build_parent_blocks(parents)
        prompt_fields = MutationPromptFields(
            count=len(parents), parent_blocks=parent_blocks
        )
        return self.user_prompt_template.format(**prompt_fields.model_dump())

    def _build_parent_blocks(self, parents: list[Program]) -> str:
        """Build formatted parent blocks for the mutation prompt."""
        blocks: list[str] = []
        for i, p in enumerate(parents):
            formatted_context = p.metadata.get(MUTATION_CONTEXT_METADATA_KEY) or ""

            block = f"""=== Parent {i + 1} ===
```python
{p.code}
```

{formatted_context}
"""
            blocks.append(block)

        return "\n\n".join(blocks)

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
        model_used = state.get("metadata", {}).get("model_used")

        if structured_output is None:
            error_msg = state.get("error", "No structured output received")
            logger.error(f"[MutationAgent] No structured output: {error_msg}")
            state["parsed_output"] = {
                "code": "",
                "structured_output": None,
                "error": error_msg,
                "model_used": model_used,
            }
            return state

        try:
            # Get code from structured output
            code_from_llm = structured_output.code

            # Fix JSON-escaped sequences from structured output serialization.
            # LLMs sometimes produce literal \n, \t, \" in the code field when
            # they confuse JSON escaping with Python syntax.
            code_from_llm = self._fix_json_escaped_code(code_from_llm)

            if state["mutation_mode"] == "diff":
                # Apply diff to parent code
                parents = state["input"]
                if len(parents) != 1:
                    raise ValueError("Diff mode requires exactly 1 parent")

                parent_code = parents[0].code
                # The code field contains the diff in diff mode
                final_code = self._apply_diff_and_extract(parent_code, code_from_llm)
            else:
                final_code = self._extract_code_block(code_from_llm)

            # Guard: reject the structured-output template echoed back as code;
            # other JSON documents are legitimate genomes (e.g. reasoning chains)
            if "def " not in final_code and final_code.lstrip().startswith("{"):
                try:
                    echoed = json.loads(final_code)
                except ValueError:
                    echoed = None
                if isinstance(echoed, dict) and {"code", "archetype"} <= echoed.keys():
                    raise ValueError(
                        "LLM echoed the structured-output template as code. "
                        f"Code starts with: {final_code[:80]!r}"
                    )

            state["final_code"] = final_code

            # Convert structured output to dict for storage
            structured_dict = structured_output.model_dump()

            citation_integrity = self._citation_integrity(
                structured_output.insight_ids_used,
                structured_output.card_ids_used,
                state.get("messages", []),
            )
            logger.info(
                "[MutationAgent] Citation integrity: insights {}/{} grounded, "
                "cards {}/{} grounded",
                citation_integrity["grounded"],
                citation_integrity["cited"],
                citation_integrity["cards_grounded"],
                citation_integrity["cards_cited"],
            )

            state["parsed_output"] = {
                "code": final_code,
                "structured_output": structured_dict,
                "archetype": structured_output.archetype,
                "justification": structured_output.justification,
                "insights_used": structured_output.insights_used,
                "insight_ids_used": [
                    c.model_dump() for c in structured_output.insight_ids_used
                ],
                "base_parent": structured_output.base_parent,
                "card_ids_used": structured_output.card_ids_used,
                "changes": structured_output.changes,
                "citation_integrity": citation_integrity,
                "model_used": model_used,
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
                "model_used": model_used,
            }

        return state

    @staticmethod
    def _citation_integrity(
        insight_ids_used: list[InsightCitation],
        card_ids_used: list[str],
        messages: list[BaseMessage],
    ) -> dict[str, int]:
        """Count cited insight and card references that were offered in the prompt.

        Standard operator: parents are labelled '=== Parent N ===' by 1-based
        number, so citations carry an int parent. Delegates to the shared,
        parent-label-agnostic grounder.
        """
        pairs = [
            (str(c.parent), c.insight)
            for c in insight_ids_used
            if c.parent > 0 and c.insight > 0
        ]
        return compute_citation_integrity(pairs, card_ids_used, messages)

    @staticmethod
    def _fix_json_escaped_code(code: str) -> str:
        """Fix JSON-escaped sequences in code from structured output.

        LLMs using structured output sometimes produce literal JSON escape
        sequences in the code field instead of the actual characters:
        - ``\\"`` instead of ``"`` (double-escaped quotes)
        - ``\\n`` instead of actual newlines (escaped newlines)
        - ``\\t`` instead of actual tabs (escaped tabs)

        This happens when the model confuses JSON string escaping with
        the Python code content. We only apply the fix when the original
        code fails to parse and the cleaned version parses successfully.
        """
        # Quick check: does code contain any JSON escape sequences?
        if "\\n" not in code and '\\"' not in code and "\\t" not in code:
            return code
        try:
            ast.parse(code)
            return code  # Already valid — don't touch it
        except SyntaxError:
            pass

        # Try unescaping JSON sequences
        cleaned = code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
        try:
            ast.parse(cleaned)
            logger.debug(
                '[MutationAgent] Fixed JSON-escaped code (\\n={}, \\t={}, \\"={})',
                code.count("\\n"),
                code.count("\\t"),
                code.count('\\"'),
            )
            return cleaned
        except SyntaxError:
            return code  # Unescaping didn't help — return original

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
