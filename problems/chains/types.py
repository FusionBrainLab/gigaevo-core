"""Chain data structures for CARL-aligned chain evolution.

Layer design
------------
**Parse layer** (gigaevo-specific)
    ``LLMStep``, ``ToolStep``, ``ToolConfig``, ``RawChainSpec`` — Pydantic
    models that parse raw ``entrypoint()`` dicts produced by LLM mutations.
    These keep ``step_type`` as a literal discriminator field and ``frozen``
    for chain evolution semantics.  Structural validation (field presence,
    types, unknown fields) lives here via Pydantic v2.

**Execution layer** (CARL)
    ``LLMStepDescription``, ``ToolStepDescription`` from ``mmar_carl`` are
    the step types stored in ``ChainSpec.steps`` and consumed by the runner.
    They carry CARL's full feature set (metrics, checkpoints, re-plan hooks,
    per-step LLM config).

The public ``validate_chain_spec`` function bridges the two layers: it parses
with ``RawChainSpec``, runs semantic validation on parse-layer objects, and
builds a ``ChainSpec`` whose steps are CARL execution types.

``PromptBuilder`` is retained as a gigaevo-specific utility used by callers
that need direct prompt assembly (e.g. offline analysis).  The step-batched
runner uses ``GigaEvoPromptTemplate`` from ``carl_bridge`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal

from mmar_carl import LLMStepDescription, ToolStepDescription
from mmar_carl.models.config import ToolStepConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    constr,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Structured field constants
# ---------------------------------------------------------------------------

STRUCTURED_FIELDS = {"aim", "stage_action", "reasoning_questions", "example_reasoning"}
METADATA_FIELDS = {"number", "title", "step_type", "dependencies", "frozen"}


# ---------------------------------------------------------------------------
# Parse-layer tool config
# ---------------------------------------------------------------------------


class ToolConfig(BaseModel):
    """Parse-layer tool step configuration with strict $-reference validation.

    Simpler than CARL's ``ToolStepConfig`` — only the fields gigaevo's LLM
    mutations produce.  The validator enforces that every ``input_mapping``
    value is a ``$``-prefixed reference string.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    input_mapping: dict[str, str]

    @model_validator(mode="after")
    def validate_dollar_refs(self) -> ToolConfig:
        for param_name, ref in self.input_mapping.items():
            if not ref.startswith("$"):
                raise ValueError(
                    f"input_mapping['{param_name}'] must be a $-reference "
                    f"string (got '{ref}')"
                )
        return self


class StopCondition(BaseModel):
    """Optional per-step early-stop condition based on step output."""

    model_config = ConfigDict(extra="forbid")

    condition_type: Literal["contains", "regex"] = "contains"
    pattern: constr(min_length=1)  # type: ignore[valid-type]
    case_sensitive: bool = False


# ---------------------------------------------------------------------------
# Parse-layer step models
# ---------------------------------------------------------------------------


def _coerce_to_str(v: object) -> str:
    """Coerce list/tuple to joined string.

    LLM mutations sometimes emit list-typed fields instead of strings.
    """
    if isinstance(v, (list, tuple)):
        return " ".join(str(item) for item in v)
    return v  # type: ignore[return-value]


class LLMStep(BaseModel):
    """Parse-layer LLM reasoning step.

    Has ``step_type: Literal["llm"]`` as a real Pydantic field so it can
    serve as the discriminator in ``RawChainSpec``'s discriminated union.
    Also carries ``frozen`` for chain evolution semantics.

    After structural + semantic validation, call ``to_carl_step()`` to obtain
    a ``mmar_carl.LLMStepDescription`` for execution.
    """

    model_config = ConfigDict(extra="forbid")

    number: int
    title: str
    step_type: Literal["llm"]
    dependencies: list[int] = Field(default_factory=list)
    frozen: bool = False

    # Required structured fields (non-empty).
    aim: constr(min_length=1)  # type: ignore[valid-type]
    stage_action: constr(min_length=1)  # type: ignore[valid-type]

    # Optional structured fields
    reasoning_questions: str = ""
    example_reasoning: str = ""
    stop_condition: StopCondition | None = None

    @field_validator(
        "aim", "stage_action", "reasoning_questions", "example_reasoning", mode="before"
    )
    @classmethod
    def _coerce_sequences_to_str(cls, v: object) -> object:
        return _coerce_to_str(v)

    def to_carl_step(self) -> LLMStepDescription:
        """Convert to a CARL ``LLMStepDescription`` for execution.

        The ``frozen`` field is parse-layer only — validation uses it before
        this conversion; the runner never needs it at execution time.
        """
        return LLMStepDescription(
            number=self.number,
            title=self.title,
            dependencies=self.dependencies,
            aim=self.aim,
            stage_action=self.stage_action,
            reasoning_questions=self.reasoning_questions,
            example_reasoning=self.example_reasoning,
        )


class ToolStep(BaseModel):
    """Parse-layer tool execution step.

    Has ``step_type: Literal["tool"]`` as a real Pydantic field for the
    discriminated union.  Carries ``frozen`` for chain evolution semantics.

    After validation, call ``to_carl_step()`` to obtain a
    ``mmar_carl.ToolStepDescription`` for execution.
    """

    model_config = ConfigDict(extra="forbid")

    number: int
    title: str
    step_type: Literal["tool"]
    dependencies: list[int] = Field(default_factory=list)
    frozen: bool = False

    step_config: ToolConfig

    def to_carl_step(self) -> ToolStepDescription:
        """Convert to a CARL ``ToolStepDescription`` for execution.

        Maps gigaevo's ``ToolConfig`` (``tool_name`` + ``input_mapping``) to
        CARL's ``ToolStepConfig``.  The ``frozen`` field is dropped — it is
        only needed during validation.
        """
        return ToolStepDescription(
            number=self.number,
            title=self.title,
            dependencies=self.dependencies,
            config=ToolStepConfig(
                tool_name=self.step_config.tool_name,
                input_mapping=self.step_config.input_mapping,
            ),
        )


# Type alias for the discriminated union used in RawChainSpec.
Step = Annotated[LLMStep | ToolStep, Field(discriminator="step_type")]


# ---------------------------------------------------------------------------
# Raw chain spec (parsed from entrypoint() output)
# ---------------------------------------------------------------------------


class RawChainSpec(BaseModel):
    """Parse-layer model for raw ``entrypoint()`` output.

    Handles structural validation: field presence, types, constraints, unknown
    field rejection, and ``$``-reference syntax in tool steps.
    """

    model_config = ConfigDict(extra="forbid")

    system_prompt: str = ""
    steps: list[Step] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Prompt builder (gigaevo-specific, kept for backward compatibility)
# ---------------------------------------------------------------------------


class PromptBuilder(BaseModel):
    """Gigaevo-specific configurable prompt assembler.

    Retained for callers that perform offline prompt assembly or need direct
    control over templates.  The step-batched runner uses
    ``GigaEvoPromptTemplate`` from ``carl_bridge`` instead.

    Templates
    ---------
    - **step_template** — formats an ``LLMStep`` into an instruction block.
      Placeholders: ``{number}``, ``{title}``, ``{aim}``, ``{stage_action}``,
      ``{reasoning_questions}``, ``{example_reasoning}``.
    - **chain_template** — wraps outer_context + step_prompt into the main
      body.  Placeholders: ``{outer_context}``, ``{step_prompt}``.
    - **history_template** — wraps history + current_task when prior steps
      exist.  Placeholders: ``{history}``, ``{current_task}``.
    - **history_entry_template** — formats a completed step's result for the
      history list.  Placeholders: ``{number}``, ``{title}``, ``{result}``.
    """

    model_config = ConfigDict(extra="forbid")

    step_template: str = (
        "Step {number}. {title}\n"
        "Objective: {aim}\n"
        "Task: {stage_action}\n"
        "Questions: {reasoning_questions}\n"
        "Example reasoning: {example_reasoning}"
    )

    chain_template: str = "Data:\n{outer_context}\n\n{step_prompt}"

    history_template: str = (
        "Previous steps:\n{history}\n\n"
        "Based on the results of previous steps, "
        "perform the following task:\n{current_task}"
    )

    history_entry_template: str = "Step {number}. {title}\nResult: {result}\n"

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
        """Assemble a complete prompt for an LLM step."""
        step_prompt = self.format_step_prompt(step)

        if visible_history:
            history_text = "\n".join(visible_history)
            step_prompt = self.history_template.format(
                history=history_text,
                current_task=step_prompt,
            )

        full_prompt = self.chain_template.format(
            outer_context=outer_context,
            step_prompt=step_prompt,
        )

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
# Runtime types
# ---------------------------------------------------------------------------


@dataclass
class ChainSpec:
    """Executable chain — validated, sorted, ready for the runner.

    ``steps`` holds CARL execution types (``LLMStepDescription`` or
    ``ToolStepDescription``) produced by ``validate_chain_spec`` after
    converting from parse-layer models.
    """

    system_prompt: str
    steps: list[LLMStepDescription | ToolStepDescription] = field(default_factory=list)
    prompt_builder: PromptBuilder = field(default_factory=PromptBuilder)


@dataclass
class ChainResult:
    """Result of running a chain on one sample.

    Keeps the compact gigaevo API used throughout all callers:

    - ``history``      — list of formatted history-entry strings (one per step)
    - ``final_output`` — raw string output of the last step
    - ``step_outputs`` — raw string outputs indexed 0-based by step position
    """

    history: list[str] = field(default_factory=list)
    final_output: str = ""
    step_outputs: list[str] = field(default_factory=list)
