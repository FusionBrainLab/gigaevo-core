"""Typed configuration schemas.

Pydantic-v2 models with ``extra='forbid'`` and ``frozen=True`` for every
shape currently expressed as YAML. Each module owns one concern; the
``experiment`` module assembles them into ``ExperimentConfig`` — the
single root the CLI loads and validates.
"""

from gigaevo.config.schemas._base import FrozenStrictModel
from gigaevo.config.schemas.algorithm import (
    AlgorithmConfig,
    ArchiveRemoverConfig,
    ArchiveSelectorConfig,
    BehaviorSpaceConfig,
    EliteSelectorConfig,
    FitnessArchiveRemoverConfig,
    FitnessProportionalEliteSelectorConfig,
    IslandConfig,
    MigrantSelectorConfig,
    MultiIslandConfig,
    SingleIslandConfig,
    SumArchiveSelectorConfig,
    TopFitnessMigrantSelectorConfig,
    WeightedEliteSelectorConfig,
)
from gigaevo.config.schemas.llm import (
    BanditRouterConfig,
    ChatOpenAIConfig,
    EnsembleRouterConfig,
    LLMConfig,
)
from gigaevo.config.schemas.redis import DataPlaneSettings, RedisConfig

__all__ = [
    "AlgorithmConfig",
    "ArchiveRemoverConfig",
    "ArchiveSelectorConfig",
    "BanditRouterConfig",
    "BehaviorSpaceConfig",
    "ChatOpenAIConfig",
    "DataPlaneSettings",
    "EliteSelectorConfig",
    "EnsembleRouterConfig",
    "FitnessArchiveRemoverConfig",
    "FitnessProportionalEliteSelectorConfig",
    "FrozenStrictModel",
    "IslandConfig",
    "LLMConfig",
    "MigrantSelectorConfig",
    "MultiIslandConfig",
    "RedisConfig",
    "SingleIslandConfig",
    "SumArchiveSelectorConfig",
    "TopFitnessMigrantSelectorConfig",
    "WeightedEliteSelectorConfig",
]
