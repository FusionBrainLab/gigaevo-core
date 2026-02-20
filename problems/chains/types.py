"""Chain data structures for CARL-aligned chain evolution.

Uses Pydantic v2 for structural validation (field presence, types, constraints,
unknown field rejection). Semantic validation (DAG, topology, frozen equality)
lives in chain_validation.py.
"""

from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, constr, model_validator


# ---------------------------------------------------------------------------
# Structured field constants (used in frozen-step comparison and elsewhere)
# ---------------------------------------------------------------------------

STRUCTURED_FIELDS = {"aim", "stage_action", "reasoning_questions", "example_reasoning"}
METADATA_FIELDS = {"number", "title", "step_type", "dependencies", "frozen"}


# ---------------------------------------------------------------------------
# Tool step nested models
# ---------------------------------------------------------------------------


class ToolConfig(BaseModel):
    """Tool step configuration with $-reference validation."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    input_mapping: dict[str, str]

    @model_validator(mode="after")
    def validate_dollar_refs(self) -> "ToolConfig":
        for param_name, ref in self.input_mapping.items():
            if not ref.startswith("$"):
                raise ValueError(
                    f"input_mapping['{param_name}'] must be a $-reference "
                    f"string (got '{ref}')"
                )
        return self


# ---------------------------------------------------------------------------
# Step models (discriminated union on step_type)
# ---------------------------------------------------------------------------


class LLMStep(BaseModel):
    """LLM reasoning step with structured fields."""

    model_config = ConfigDict(extra="forbid")

    number: int
    title: str
    step_type: Literal["llm"]
    dependencies: list[int] = Field(default_factory=list)
    frozen: bool = False

    # Required structured fields (non-empty)
    aim: constr(min_length=1)  # type: ignore[valid-type]
    stage_action: constr(min_length=1)  # type: ignore[valid-type]

    # Optional structured fields
    reasoning_questions: str = ""
    example_reasoning: str = ""


class ToolStep(BaseModel):
    """Tool execution step with step_config."""

    model_config = ConfigDict(extra="forbid")

    number: int
    title: str
    step_type: Literal["tool"]
    dependencies: list[int] = Field(default_factory=list)
    frozen: bool = False

    step_config: ToolConfig


Step = Annotated[LLMStep | ToolStep, Field(discriminator="step_type")]


# ---------------------------------------------------------------------------
# Raw chain spec (parsed from entrypoint() output)
# ---------------------------------------------------------------------------


class RawChainSpec(BaseModel):
    """Pydantic model for parsing raw entrypoint() output.

    Handles all structural validation: field presence, types, constraints,
    unknown field rejection, $-reference syntax.
    """

    model_config = ConfigDict(extra="forbid")

    system_prompt: str = ""
    steps: list[Step] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Prompt builder (CARL-style configurable prompt assembler)
# ---------------------------------------------------------------------------


class PromptBuilder(BaseModel):
    """CARL-style configurable prompt assembler.

    - **step_template** — formats an LLMStep into an instruction block.
      Available placeholders: ``{number}``, ``{title}``, ``{aim}``,
      ``{stage_action}``, ``{reasoning_questions}``,
      ``{example_reasoning}``.
    - **chain_template** — wraps outer_context + step_prompt into the
      main body.  Available: ``{outer_context}``, ``{step_prompt}``.
      "System Instructions:" prefix is prepended automatically when
      a system prompt is provided (like CARL's ``format_chain_prompt``).
    - **history_template** — wraps history + current_task when prior steps
      exist.  Available: ``{history}``, ``{current_task}``.
    - **history_entry_template** — formats a completed step's result for
      the history list.
      Available: ``{number}``, ``{title}``, ``{result}``.
    """

    model_config = ConfigDict(extra="forbid")

    step_template: str = (
        "Step {number}. {title}\n"
        "Objective: {aim}\n"
        "Task: {stage_action}\n"
        "Questions: {reasoning_questions}\n"
        "Example reasoning: {example_reasoning}"
    )

    chain_template: str = (
        "Data:\n{outer_context}\n\n"
        "{step_prompt}"
    )

    history_template: str = (
        "Previous steps:\n{history}\n\n"
        "Based on the results of previous steps, "
        "perform the following task:\n{current_task}"
    )

    history_entry_template: str = "Step {number}. {title}\nResult: {result}\n"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_step_prompt(self, step: LLMStep) -> str:
        """Format an LLM step into an instruction block."""
        return self.step_template.format(
            number=step.number,
            title=step.title,
            aim=step.aim,
            stage_action=step.stage_action,
            reasoning_questions=step.reasoning_questions,
            example_reasoning=step.example_reasoning,
        )

    def build_prompt(
        self,
        step: LLMStep,
        visible_history: list[str],
        outer_context: str,
        system_prompt: str,
    ) -> str:
        """Assemble a complete prompt for an LLM step.

        Flow (mirrors CARL's ``format_chain_prompt``):
        1. Render the step instruction block via ``format_step_prompt``.
        2. If there is history, wrap with ``history_template``.
        3. Wrap with ``chain_template``.
        4. Prepend ``System Instructions:`` if system_prompt is non-empty.
        """
        step_prompt = self.format_step_prompt(step)

        # Wrap with history if previous steps exist
        if visible_history:
            history_text = "\n".join(visible_history)
            step_prompt = self.history_template.format(
                history=history_text,
                current_task=step_prompt,
            )

        # Wrap with chain template
        full_prompt = self.chain_template.format(
            outer_context=outer_context,
            step_prompt=step_prompt,
        )

        # Prepend system instructions (like CARL's format_chain_prompt)
        if system_prompt and system_prompt.strip():
            return f"System Instructions:\n{system_prompt}\n\n{full_prompt}"

        return full_prompt

    def format_history_entry(
        self,
        number: int,
        title: str,
        result: str,
    ) -> str:
        """Format a completed step result for the history list."""
        return self.history_entry_template.format(
            number=number,
            title=title,
            result=result,
        )


# ---------------------------------------------------------------------------
# Runtime types (post-validation, used by runner)
# ---------------------------------------------------------------------------


@dataclass
class ChainSpec:
    """Executable chain — validated, sorted, ready for runner."""

    system_prompt: str
    steps: list[LLMStep | ToolStep] = field(default_factory=list)
    prompt_builder: PromptBuilder = field(default_factory=PromptBuilder)


@dataclass
class ChainResult:
    """Result of running a chain on one sample."""

    history: list[str] = field(default_factory=list)
    final_output: str = ""
    step_outputs: list[str] = field(default_factory=list)
