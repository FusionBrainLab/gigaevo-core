"""GigaEvo validator for FeatureGraph genomes scored by TabM."""

from __future__ import annotations

from problems.tabular_dag_baselines.tabm.tabm_backend import (
    FeatureGraphModel,
    TabMConfig,
    effective_amp_dtype,
)
from problems.tabular_dag_baselines.validation import (
    score_payload_on_test,
    validate_payload,
)


def _builder(config: TabMConfig):
    return lambda graph, device: FeatureGraphModel(graph, device=device, config=config)


def validate(payload):
    """Validate a FeatureGraph and score it with fixed TabM-mini+PLE."""
    config = TabMConfig.from_env()
    return validate_payload(
        payload,
        estimator_name=f"{config.arch_type}-ple",
        model_builder=_builder(config),
        config=config,
        gpu_model="tabm",
        artifact_extra={
            "tabm_k": config.k,
            "tabm_refit": config.refit,
            "tabm_share_training_batches": config.share_training_batches,
        },
        device_artifact_builder=lambda device: {
            "tabm_amp_dtype": effective_amp_dtype(device or "cpu", enabled=config.amp)
        },
    )


def score_on_test(payload):
    """Score on the untouched test split with the shared tabular protocol."""
    config = TabMConfig.from_env()
    return score_payload_on_test(
        payload,
        model_builder=_builder(config),
        gpu_model="tabm",
    )
