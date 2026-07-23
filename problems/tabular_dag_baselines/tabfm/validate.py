"""GigaEvo validator for FeatureGraph genomes scored by TabFM 1.0."""

from problems.tabular._common import tabular_data
from problems.tabular_dag_baselines.tabfm.backend import (
    TABFM_MAX_CLASSES,
    FeatureGraphModel,
    ModelCache,
    TabFMConfig,
    TabFMModelType,
    ensure_tabfm_ready,
)
from problems.tabular_dag_baselines.validation import (
    score_payload_on_test,
    validate_payload,
)


def _runtime(config: TabFMConfig):
    model_cache: ModelCache = {}

    def builder(graph, device):
        return FeatureGraphModel(
            graph,
            device=device,
            config=config,
            model_cache=model_cache,
        )

    return builder, model_cache.clear


def _readiness_check(config: TabFMConfig):
    def check(_graph, dataset):
        model_type: TabFMModelType = (
            "regression"
            if dataset.task_type == tabular_data.REGRESSION
            else "classification"
        )
        if (
            model_type == "classification"
            and dataset.n_classes is not None
            and dataset.n_classes > TABFM_MAX_CLASSES
        ):
            raise ValueError(
                f"TabFM supports at most {TABFM_MAX_CLASSES} classes; "
                f"dataset declares {dataset.n_classes}"
            )
        ensure_tabfm_ready(config, model_type=model_type)

    return check


def validate(payload):
    config = TabFMConfig.from_env()
    builder, cleanup = _runtime(config)
    return validate_payload(
        payload,
        estimator_name="tabfm-1.0-pytorch",
        model_builder=builder,
        config=config,
        gpu_model="tabfm",
        readiness_check=_readiness_check(config),
        resource_cleanup=cleanup,
    )


def score_on_test(payload):
    config = TabFMConfig.from_env()
    builder, cleanup = _runtime(config)
    return score_payload_on_test(
        payload,
        model_builder=builder,
        gpu_model="tabfm",
        readiness_check=_readiness_check(config),
        resource_cleanup=cleanup,
    )
