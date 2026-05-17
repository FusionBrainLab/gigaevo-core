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
from gigaevo.config.schemas.engine import (
    AcceptorConfig,
    AllCombinationsParentSelectorConfig,
    EngineConfig,
    GenerationalEngineConfig,
    ParentSelectorConfig,
    RandomParentSelectorConfig,
    StandardAcceptorConfig,
    SteadyStateEngineConfig,
)
from gigaevo.config.schemas.pipeline import (
    AlgoTuneSpeedPipelineBuilderConfig,
    AutoPipelineBuilderConfig,
    CMAOptPipelineBuilderConfig,
    ContextPipelineBuilderConfig,
    DefaultPipelineBuilderConfig,
    OptunaOptPipelineBuilderConfig,
    PipelineBuilderConfig,
    PipelineConfig,
    StructuralMetricsPipelineBuilderConfig,
)
from gigaevo.config.schemas.llm import (
    BanditRouterConfig,
    ChatOpenAIConfig,
    EnsembleRouterConfig,
    LLMConfig,
)
from gigaevo.config.schemas.logging import (
    LoggingConfig,
    LoggingSettings,
    RedisMetricsTrackerConfig,
    TBTrackerConfig,
    TrackerConfig,
    WandBTrackerConfig,
)
from gigaevo.config.schemas.problem import ProblemConfig
from gigaevo.config.schemas.prompt import (
    FixedDirPromptFetcherConfig,
    GigaEvoArchivePromptFetcherConfig,
    PromptFetcherConfig,
)
from gigaevo.config.schemas.scheduling import (
    ChainFeatureExtractorConfig,
    CodeFeatureExtractorConfig,
    FIFOConfig,
    FeatureExtractorConfig,
    LPTConfig,
    PredictorConfig,
    RidgePredictorConfig,
    SchedulingConfig,
    SimpleHeuristicPredictorConfig,
)
from gigaevo.config.schemas.redis import DataPlaneSettings, RedisConfig
from gigaevo.config.schemas.experiment import ExperimentConfig

__all__ = [
    "AcceptorConfig",
    "AlgorithmConfig",
    "AlgoTuneSpeedPipelineBuilderConfig",
    "AllCombinationsParentSelectorConfig",
    "ArchiveRemoverConfig",
    "ArchiveSelectorConfig",
    "AutoPipelineBuilderConfig",
    "BanditRouterConfig",
    "BehaviorSpaceConfig",
    "CMAOptPipelineBuilderConfig",
    "ChainFeatureExtractorConfig",
    "ChatOpenAIConfig",
    "CodeFeatureExtractorConfig",
    "ContextPipelineBuilderConfig",
    "DataPlaneSettings",
    "DefaultPipelineBuilderConfig",
    "EliteSelectorConfig",
    "EngineConfig",
    "EnsembleRouterConfig",
    "ExperimentConfig",
    "FIFOConfig",
    "FeatureExtractorConfig",
    "FixedDirPromptFetcherConfig",
    "FitnessArchiveRemoverConfig",
    "FitnessProportionalEliteSelectorConfig",
    "FrozenStrictModel",
    "GenerationalEngineConfig",
    "GigaEvoArchivePromptFetcherConfig",
    "IslandConfig",
    "LLMConfig",
    "LPTConfig",
    "LoggingConfig",
    "LoggingSettings",
    "MigrantSelectorConfig",
    "MultiIslandConfig",
    "OptunaOptPipelineBuilderConfig",
    "ParentSelectorConfig",
    "PipelineBuilderConfig",
    "PipelineConfig",
    "PredictorConfig",
    "ProblemConfig",
    "PromptFetcherConfig",
    "RandomParentSelectorConfig",
    "RedisConfig",
    "RedisMetricsTrackerConfig",
    "RidgePredictorConfig",
    "SchedulingConfig",
    "SimpleHeuristicPredictorConfig",
    "SingleIslandConfig",
    "StandardAcceptorConfig",
    "SteadyStateEngineConfig",
    "StructuralMetricsPipelineBuilderConfig",
    "SumArchiveSelectorConfig",
    "TBTrackerConfig",
    "TopFitnessMigrantSelectorConfig",
    "TrackerConfig",
    "WandBTrackerConfig",
    "WeightedEliteSelectorConfig",
]
