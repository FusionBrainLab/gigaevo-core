"""GigaEvo validator for FeatureGraph genomes scored by TabICLv2."""

from problems.tabular_dag_baselines.tabicl.backend import (
    FeatureGraphModel,
    TabICLConfig,
)
from problems.tabular_dag_baselines.validation import (
    score_payload_on_test,
    validate_payload,
)


def _builder(config: TabICLConfig):
    return lambda graph, device: FeatureGraphModel(graph, device=device, config=config)


def validate(payload):
    config = TabICLConfig.from_env()
    return validate_payload(
        payload,
        estimator_name="tabiclv2",
        model_builder=_builder(config),
        config=config,
        gpu_model="tabicl",
    )


def score_on_test(payload):
    config = TabICLConfig.from_env()
    return score_payload_on_test(
        payload,
        model_builder=_builder(config),
        gpu_model="tabicl",
    )
