"""Stages for the prompt evolution pipeline.

PromptExecutionStage: executes entrypoint() from the program to get prompt text.
PromptFitnessStage: reads mutation success stats from the main run's Redis.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from gigaevo.programs.core_types import StageIO, VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import FloatDictContainer
from gigaevo.programs.stages.stage_registry import StageRegistry
from gigaevo.prompts.coevolution.stats import PromptStatsProvider, prompt_text_to_id


class PromptExecutionOutput(StageIO):
    """Output of PromptExecutionStage."""

    prompt_text: str
    prompt_id: str


@StageRegistry.register(
    description="Execute prompt program's entrypoint() to get prompt text"
)
class PromptExecutionStage(Stage):
    """Executes entrypoint() from the program code in a clean namespace.

    The program's entrypoint() must return a str (the mutation system prompt text).
    Stores the prompt_text and prompt_id (sha256[:16] of the text) on the stage output.
    """

    InputsModel = VoidInput
    OutputModel = PromptExecutionOutput

    def __init__(self, *, timeout: float = 30.0, **kwargs):
        super().__init__(timeout=timeout, **kwargs)

    async def compute(self, program: Program) -> PromptExecutionOutput:
        code = program.code
        namespace: dict[str, Any] = {}
        try:
            exec(compile(code, "<prompt_program>", "exec"), namespace)  # noqa: S102
        except SyntaxError as exc:
            raise ValueError(f"Prompt program has syntax error: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Prompt program failed to compile/exec: {exc}") from exc

        entrypoint_fn = namespace.get("entrypoint")
        if not callable(entrypoint_fn):
            raise ValueError("Prompt program has no callable entrypoint() function")

        try:
            result = entrypoint_fn()
        except Exception as exc:
            raise ValueError(f"entrypoint() raised an exception: {exc}") from exc

        if not isinstance(result, str):
            raise ValueError(
                f"entrypoint() must return str, got {type(result).__name__}"
            )
        if not result.strip():
            raise ValueError("entrypoint() returned empty string")

        prompt_id = prompt_text_to_id(result)
        logger.debug(
            f"[PromptExecutionStage] Executed entrypoint(): "
            f"{len(result)} chars, id={prompt_id}"
        )
        return PromptExecutionOutput(prompt_text=result, prompt_id=prompt_id)


class PromptFitnessInputs(StageIO):
    """Inputs for PromptFitnessStage."""

    execution_output: PromptExecutionOutput


@StageRegistry.register(
    description="Evaluate prompt fitness from mutation success rate"
)
class PromptFitnessStage(Stage):
    """Evaluates prompt fitness from mutation success rate in the main run.

    Reads per-prompt stats (trials, successes) written by the main run's
    GigaEvoArchivePromptFetcher.record_outcome(). Returns:
      - fitness: success_rate (0.0 if insufficient trials)
      - is_valid: 1.0
      - prompt_length: float length of the prompt text (behavior dimension)

    The stats_provider is injected via constructor — no global state.

    Args:
        stats_provider: Provides per-prompt stats from the main run's Redis
        min_trials: Minimum trials before reporting real success rate
    """

    InputsModel = PromptFitnessInputs
    OutputModel = FloatDictContainer

    def __init__(
        self,
        stats_provider: PromptStatsProvider,
        min_trials: int = 5,
        timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(timeout=timeout, **kwargs)
        self._stats_provider = stats_provider
        self._min_trials = min_trials

    async def compute(self, program: Program) -> FloatDictContainer:
        execution_output: PromptExecutionOutput = self.params.execution_output

        prompt_id = execution_output.prompt_id
        stats = await self._stats_provider.get_stats(prompt_id)

        fitness = stats.success_rate
        prompt_length = float(len(execution_output.prompt_text))

        logger.debug(
            f"[PromptFitnessStage] prompt_id={prompt_id} "
            f"trials={stats.trials} successes={stats.successes} "
            f"fitness={fitness:.4f}"
        )

        metrics = {
            "fitness": fitness,
            "is_valid": 1.0,
            "prompt_length": prompt_length,
        }
        program.add_metrics(metrics)
        return FloatDictContainer(data=metrics)
