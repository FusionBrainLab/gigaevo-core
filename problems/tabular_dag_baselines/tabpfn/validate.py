"""GigaEvo validator for FeatureGraph genomes scored by TabPFN v3."""

from problems.tabular._common import tabular_data
from problems.tabular_dag_baselines.tabpfn.backend import (
    FeatureGraphModel,
    TabPFNConfig,
    ensure_tabpfn_ready,
)
from problems.tabular_dag_baselines.validation import (
    score_payload_on_test,
    validate_payload,
)


def _builder(config: TabPFNConfig):
    return lambda graph, device: FeatureGraphModel(graph, device=device, config=config)


def _readiness_check(config: TabPFNConfig):
    def check(_graph, dataset):
        which = (
            "regressor"
            if dataset.task_type == tabular_data.REGRESSION
            else "classifier"
        )
        ensure_tabpfn_ready(config, which=which)

    return check


def validate(payload):
    config = TabPFNConfig.from_env()
    return validate_payload(
        payload,
        estimator_name="tabpfn-v3",
        model_builder=_builder(config),
        config=config,
        gpu_model="tabpfn",
        readiness_check=_readiness_check(config),
    )


def score_on_test(payload):
    config = TabPFNConfig.from_env()
    return score_payload_on_test(
        payload,
        model_builder=_builder(config),
        gpu_model="tabpfn",
        readiness_check=_readiness_check(config),
    )
