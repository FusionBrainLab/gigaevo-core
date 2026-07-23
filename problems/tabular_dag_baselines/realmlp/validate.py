"""GigaEvo validator for FeatureGraph genomes scored by RealMLP-TD."""

from problems.tabular_dag_baselines.realmlp.backend import (
    FeatureGraphModel,
    RealMLPConfig,
)
from problems.tabular_dag_baselines.validation import (
    score_payload_on_test,
    validate_payload,
)


def _builder(config: RealMLPConfig):
    return lambda graph, device: FeatureGraphModel(graph, device=device, config=config)


def validate(payload):
    config = RealMLPConfig.from_env()
    return validate_payload(
        payload,
        estimator_name="realmlp-td",
        model_builder=_builder(config),
        config=config,
        gpu_model="realmlp",
    )


def score_on_test(payload):
    config = RealMLPConfig.from_env()
    return score_payload_on_test(
        payload,
        model_builder=_builder(config),
        gpu_model="realmlp",
    )
