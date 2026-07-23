"""GigaEvo validator for FeatureGraph genomes scored by XGBoost."""

from problems.tabular_dag_baselines.validation import (
    score_payload_on_test,
    validate_payload,
)
from problems.tabular_dag_baselines.xgboost.backend import (
    FeatureGraphModel,
    XGBoostConfig,
)


def _builder(config: XGBoostConfig):
    return lambda graph, _device: FeatureGraphModel(graph, config=config)


def validate(payload):
    config = XGBoostConfig.from_env()
    return validate_payload(
        payload,
        estimator_name="xgboost",
        model_builder=_builder(config),
        config=config,
    )


def score_on_test(payload):
    config = XGBoostConfig.from_env()
    return score_payload_on_test(payload, model_builder=_builder(config))
