"""CARL-integrated chain infrastructure for GigaEvo.

Modules
-------
- ``types``: Parse-layer and runtime Pydantic models
- ``carl_bridge``: Adapters for CARL integration
- ``chain_validation``: Semantic validation and CARL type conversion
- ``chain_runner``: Step-batched execution engine with CARL backend
- ``runner_config``: Feedback modes and execution modes (RunnerConfig)
"""

from problems.chains.runner_config import (
    DatasetFeedbackConfig,
    FeedbackMode,
    MetricFeedbackConfig,
    RunnerConfig,
    SelfCriticConfig,
    SimpleRetryConfig,
    StepExecutionMode,
)
