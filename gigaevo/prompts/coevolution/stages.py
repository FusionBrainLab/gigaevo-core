"""Stages for the prompt evolution pipeline.

PromptExecutionStage: executes entrypoint() from the program to get prompt text.
PromptFitnessStage: reads mutation success stats from the main run's Redis.
PromptInsightsStage / PromptLineageStage: cache-aware wrappers that invalidate
    when fitness changes and skip when fitness is a dummy Beta prior.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from gigaevo.programs.core_types import ProgramStageResult, StageIO, VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import FloatDictContainer
from gigaevo.programs.stages.insights import InsightsOutput, InsightsStage
from gigaevo.programs.stages.insights_lineage import LineageAnalysesOutput, LineageStage
from gigaevo.programs.stages.stage_registry import StageRegistry
from gigaevo.prompts.coevolution.stats import PromptStatsProvider, prompt_text_to_id


class PromptExecutionOutput(StageIO):
    """Output of PromptExecutionStage.

    prompt_text: system prompt template (required)
    user_text: user prompt template (None if entrypoint() returns str, not dict)
    prompt_id: sha256[:16] of system prompt text
    """

    prompt_text: str
    user_text: str | None = None
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

    async def compute(self, program: Program) -> PromptExecutionOutput:  # type: ignore[override]
        code = program.code
        if "def entrypoint" not in code:
            raise ValueError(
                "Prompt program must contain 'def entrypoint()'. "
                "Got non-Python content (possibly JSON template). "
                f"Code starts with: {code[:80]!r}"
            )
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

        if isinstance(result, str):
            if not result.strip():
                raise ValueError("entrypoint() returned empty string")
            system_text = result
            user_text = None
        elif isinstance(result, dict):
            system_text = result.get("system", "")
            if not isinstance(system_text, str) or not system_text.strip():
                raise ValueError(
                    "dict entrypoint() must have a non-empty 'system' key (str)"
                )
            user_text = result.get("user")
            if user_text is not None and (
                not isinstance(user_text, str) or not user_text.strip()
            ):
                raise ValueError(
                    "dict entrypoint() 'user' key must be a non-empty str when present"
                )
        else:
            raise ValueError(
                f"entrypoint() must return str or dict, got {type(result).__name__}"
            )

        prompt_id = prompt_text_to_id(system_text, user_text=user_text)
        logger.debug(
            f"[PromptExecutionStage] Executed entrypoint(): "
            f"system={len(system_text)} chars, user={len(user_text) if user_text else 0} chars, "
            f"id={prompt_id}"
        )
        return PromptExecutionOutput(
            prompt_text=system_text, user_text=user_text, prompt_id=prompt_id
        )


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

    Uses NeverCached because fitness depends on external Redis stats that
    change as the main run progresses — inputs (prompt text) stay the same
    but the stats they reference do not.

    Args:
        stats_provider: Provides per-prompt stats from the main run's Redis
        min_trials: Minimum trials before reporting real success rate
    """

    InputsModel = PromptFitnessInputs
    OutputModel = FloatDictContainer
    cache_handler = NO_CACHE

    def __init__(
        self,
        stats_provider: PromptStatsProvider,
        prior_alpha: float = 1.0,
        prior_beta: float = 3.0,
        timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(timeout=timeout, **kwargs)
        self._stats_provider = stats_provider
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

    async def compute(self, program: Program) -> FloatDictContainer:  # type: ignore[override]
        execution_output: PromptExecutionOutput = self.params.execution_output  # type: ignore[attr-defined]

        prompt_id = execution_output.prompt_id
        stats = await self._stats_provider.get_stats(prompt_id)

        # Bayesian posterior mean with Beta(alpha, beta) prior.
        # Default Beta(1,3): untested prompts start at 0.25, preventing
        # them from outranking mediocre-but-tested prompts.
        fitness = (stats.successes + self._prior_alpha) / (
            stats.trials + self._prior_alpha + self._prior_beta
        )
        prompt_length = float(len(execution_output.prompt_text))

        logger.debug(
            f"[PromptFitnessStage] prompt_id={prompt_id} "
            f"trials={stats.trials} successes={stats.successes} "
            f"fitness={fitness:.4f} mean_child_f1={stats.mean_child_fitness:.4f} "
            f"(prior=Beta({self._prior_alpha},{self._prior_beta}))"
        )

        metrics: dict[str, float] = {
            "fitness": fitness,
            "is_valid": 1.0,
            "prompt_length": prompt_length,
            "trials": float(stats.trials),
            "successes": float(stats.successes),
            "mean_child_fitness": stats.mean_child_fitness,
        }
        # Store per-metric means from main runs (e.g., EM, F1, retrieval scores)
        if stats.mean_metrics:
            for k, v in stats.mean_metrics.items():
                metrics[f"main_{k}"] = v

        program.add_metrics(metrics)
        return FloatDictContainer(data=metrics)


# ---------------------------------------------------------------------------
# Cache-aware wrappers for InsightsStage / LineageStage
# ---------------------------------------------------------------------------
# In the prompt evolution pipeline, fitness starts as a Beta(1,3) prior
# (0.25, trials=0) and updates as the main runs report mutation outcomes.
# The base InsightsStage / LineageStage use InputHashCache with VoidInput,
# so their cache key never changes — they compute once at dummy fitness and
# never re-run when real data arrives.
#
# These subclasses accept the validated metrics dict as an optional input.
# Including it in the InputsModel means the cache key changes whenever
# trials/fitness change, causing automatic re-computation.  When trials=0
# (no real data), they skip entirely to avoid generating misleading
# insights from the Beta prior.
# ---------------------------------------------------------------------------


class FitnessMetricsInput(StageIO):
    """Optional fitness metrics for cache invalidation in prompt evolution."""

    fitness_metrics: FloatDictContainer | None = None


@StageRegistry.register(
    description="LLM insights for a prompt program (cache-invalidates on fitness change)"
)
class PromptInsightsStage(InsightsStage):
    """InsightsStage that invalidates cache when prompt fitness changes.

    Skips when trials=0 (Beta prior, no real data) to avoid misleading
    insights like "this prompt is bad" based on a dummy 0.25 fitness.
    """

    InputsModel = FitnessMetricsInput  # type: ignore[assignment]

    async def compute(self, program: Program) -> InsightsOutput | ProgramStageResult:  # type: ignore[override]
        fm = self.params.fitness_metrics  # type: ignore[attr-defined]
        if fm is not None and fm.data.get("trials", 0) == 0:
            return ProgramStageResult.skipped(
                message="No trials yet — fitness is Beta prior, insights deferred",
                stage=self.stage_name,
            )
        return await super().compute(program)


@StageRegistry.register(
    description="Lineage analysis for a prompt program (cache-invalidates on fitness change)"
)
class PromptLineageStage(LineageStage):
    """LineageStage that invalidates cache when prompt fitness changes.

    Same rationale as PromptInsightsStage: skip when trials=0 to avoid
    misleading transition analysis based on dummy fitness values.
    """

    InputsModel = FitnessMetricsInput  # type: ignore[assignment]

    async def compute(
        self, program: Program
    ) -> LineageAnalysesOutput | ProgramStageResult:  # type: ignore[override]
        fm = self.params.fitness_metrics  # type: ignore[attr-defined]
        if fm is not None and fm.data.get("trials", 0) == 0:
            return ProgramStageResult.skipped(
                message="No trials yet — fitness is Beta prior, lineage deferred",
                stage=self.stage_name,
            )
        result = await super().compute(program)
        return result  # type: ignore[return-value]
