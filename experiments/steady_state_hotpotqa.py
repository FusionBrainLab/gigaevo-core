"""Reference experiment: steady-state MAP-Elites on HotpotQA static F1.

Demonstrates the canonical pattern every shipped experiment follows:
a single ``build() -> ExperimentConfig`` returning a fully-validated,
frozen configuration tree.

Variants that touch a single scalar can use ``model_copy(update=...)``::

    from experiments.steady_state_hotpotqa import build as base
    seven_seed = base().model_copy(update={"seed": 7})

``model_copy`` shallow-copies without re-validating, so updates that
would violate a cross-field invariant (for example renaming the
experiment without rewriting ``dataplane.key_prefix``) must round-trip
through ``ExperimentConfig.model_validate({**base().model_dump(),
**update})`` to trigger the validators.
"""

from __future__ import annotations

from pathlib import Path

from gigaevo.config.schemas import (
    BanditRouterConfig,
    BehaviorSpaceConfig,
    ChatOpenAIConfig,
    ContextPipelineBuilderConfig,
    DataPlaneSettings,
    ExperimentConfig,
    FitnessArchiveRemoverConfig,
    FitnessProportionalEliteSelectorConfig,
    IslandConfig,
    PipelineConfig,
    ProblemConfig,
    RedisConfig,
    SingleIslandConfig,
    SteadyStateEngineConfig,
    SumArchiveSelectorConfig,
    TopFitnessMigrantSelectorConfig,
)

_PROBLEM_DIR = Path("problems/chains/hotpotqa/static_f1")
_NAME = "steady_state_hotpotqa"


def build() -> ExperimentConfig:
    redis = RedisConfig(host="localhost", port=6379, db=0)
    fitness_behavior_space = BehaviorSpaceConfig(
        keys=["fitness"],
        bounds=[(0.0, 1.0)],
        resolutions=[150],
        binning_types=["linear"],
        dynamic=True,
    )
    main_island = IslandConfig(
        island_id="main",
        max_size=300,
        behavior_space=fitness_behavior_space,
        archive_selector=SumArchiveSelectorConfig(
            fitness_keys=["fitness"],
            fitness_key_higher_is_better=[True],
        ),
        elite_selector=FitnessProportionalEliteSelectorConfig(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
        ),
        archive_remover=FitnessArchiveRemoverConfig(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
        ),
        migrant_selector=TopFitnessMigrantSelectorConfig(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
        ),
    )
    return ExperimentConfig(
        name=_NAME,
        seed=50,
        redis=redis,
        dataplane=DataPlaneSettings(redis=redis, key_prefix=f"gigaevo:{_NAME}"),
        problem=ProblemConfig(name=_NAME, problem_dir=_PROBLEM_DIR),
        algorithm=SingleIslandConfig(island=main_island),
        engine=SteadyStateEngineConfig(
            max_generations=200,
            max_elites_per_generation=10,
            max_mutations_per_generation=50,
            max_in_flight=5,
        ),
        pipeline=PipelineConfig(builder=ContextPipelineBuilderConfig()),
        llm=BanditRouterConfig(
            models=[
                ChatOpenAIConfig(model="gpt-4o-mini", temperature=0.7),
                ChatOpenAIConfig(
                    model="deepseek/deepseek-chat",
                    temperature=0.7,
                    base_url="https://openrouter.ai/api/v1",
                ),
            ],
            exploration_constant=1.41,
            window_size=100,
        ),
    )
