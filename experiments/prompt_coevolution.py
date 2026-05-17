"""Prompt co-evolution experiment matching
``config/experiment/prompt_coevolution.yaml``.

The main run uses a GigaEvoArchivePromptFetcher that pulls system
prompts from a paired prompt-evolution run. The shipped YAML
documents the two-run dance: this is the main run; the paired
prompt run separately optimises mutation prompts.

Required: ``prompt_redis_db`` of the paired prompt run (default 6
matching the YAML example), and the main_redis_db (this run's DB)
for the prompt run to read outcome stats from.
"""

from __future__ import annotations

from gigaevo.config.algorithm_presets import build_single_island
from gigaevo.config.engine_presets import build_generational
from gigaevo.config.llm_presets import build_single
from gigaevo.config.pipeline_presets import build_auto
from gigaevo.config.problem_presets import build_heilbron
from gigaevo.config.runner_presets import build_default_runner
from gigaevo.config.schemas import (
    DataPlaneSettings,
    ExperimentConfig,
    GigaEvoArchivePromptFetcherConfig,
    RedisConfig,
)


_NAME = "heilbron_prompt_coevolution"


def build() -> ExperimentConfig:
    redis = RedisConfig()
    # The coevolved prompt fetcher targets a paired prompt-evolution
    # run on Redis DB 6 (YAML example). main_redis_prefix matches
    # the YAML's ${problem.name} substitution so the prompt run can
    # write outcome stats back to a stable key namespace.
    prompt_fetcher = GigaEvoArchivePromptFetcherConfig(
        prompt_redis_db=6,
        main_redis_prefix="heilbron",
        main_redis_db=0,
    )
    return ExperimentConfig(
        name=_NAME,
        seed=42,
        redis=redis,
        dataplane=DataPlaneSettings(redis=redis, key_prefix=f"gigaevo:{_NAME}"),
        problem=build_heilbron(name=_NAME),
        algorithm=build_single_island(),
        engine=build_generational(),
        pipeline=build_auto(),
        llm=build_single(
            "google/gemini-3-flash-preview",
            base_url="https://openrouter.ai/api/v1",
        ),
        runner=build_default_runner(),
        prompt_fetcher=prompt_fetcher,
    )
