"""Prompt co-evolution -- main-run side.

The main run uses a ``GigaEvoArchivePromptFetcher`` that pulls system
prompts from a paired prompt-evolution run. This file describes the
main run; the paired prompt run separately optimises mutation prompts
and lives in its own experiment file.

Required: ``prompt_redis_db`` of the paired prompt run (default 6),
and the main-run Redis DB so the prompt run can read outcome stats
back.
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
    # run on Redis DB 6. ``main_redis_prefix`` matches the problem
    # name so the prompt run writes outcome stats back to a stable
    # key namespace; ``main_redis_db`` mirrors the main run's DB so
    # an override flows through both sides of the pairing.
    prompt_fetcher = GigaEvoArchivePromptFetcherConfig(
        prompt_redis_db=6,
        main_redis_prefix="heilbron",
        main_redis_db=redis.db,
    )
    return ExperimentConfig(
        name=_NAME,
        seed=47,
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
