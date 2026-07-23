"""GigaEvo validator for FeatureGraph genomes scored by LightGBM."""

from problems.tabular_dag_baselines.lightgbm.backend import (
    FeatureGraphModel,
    LightGBMConfig,
)
from problems.tabular_dag_baselines.validation import (
    score_payload_on_test,
    validate_payload,
)


def _builder(config: LightGBMConfig):
    return lambda graph, _device: FeatureGraphModel(graph, config=config)


def validate(payload):
    config = LightGBMConfig.from_env()
    return validate_payload(
        payload,
        estimator_name="lightgbm",
        model_builder=_builder(config),
        config=config,
    )


def score_on_test(payload):
    config = LightGBMConfig.from_env()
    return score_payload_on_test(payload, model_builder=_builder(config))
