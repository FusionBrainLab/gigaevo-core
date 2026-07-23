"""Common validator shell; model packages supply only their estimator factory."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

_PROBLEMS_DIR = Path(__file__).resolve().parent.parent
_TABULAR_COMMON = _PROBLEMS_DIR / "tabular" / "_common"
if str(_TABULAR_COMMON) not in sys.path:
    sys.path.insert(0, str(_TABULAR_COMMON))

import tabular_data  # noqa: E402
from tabular_problem import build  # noqa: E402

from problems.dag_tab.execution import (  # noqa: E402
    assert_split_invariant,
    assert_target_round_trip,
)
from problems.dag_tab.graph import FeatureGraph  # noqa: E402
from problems.dag_tab.validate import (  # noqa: E402
    _INVALID,
    _failure_artifact,
    _frame,
)
from problems.tabular_dag_baselines.gpu_pool import (  # noqa: E402
    random_gpu_lease,
    release_cuda,
)

ModelBuilder = Callable[[FeatureGraph, str | None], object]
DeviceArtifactBuilder = Callable[[str | None], dict[str, Any]]
ReadinessCheck = Callable[[FeatureGraph, object], None]
ResourceCleanup = Callable[[], None]


def _config_payload(config: object) -> object:
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return config


def validate_payload(
    payload: object,
    *,
    estimator_name: str,
    model_builder: ModelBuilder,
    config: object,
    gpu_model: str | None = None,
    artifact_extra: dict[str, Any] | None = None,
    device_artifact_builder: DeviceArtifactBuilder | None = None,
    readiness_check: ReadinessCheck | None = None,
    resource_cleanup: ResourceCleanup | None = None,
):
    """Run the canonical graph checks and tabular CV around one estimator."""

    stage = "schema"
    try:
        graph = FeatureGraph.model_validate(payload)
        stage = "dataset_contract"
        dataset = tabular_data.load_dataset(graph.dataset)
        expected = [f"x{i}" for i in range(dataset.X_train.shape[1])]
        if graph.raw_columns != expected:
            raise ValueError(
                f"raw_columns must exactly match dataset columns {expected}; "
                f"got {graph.raw_columns}"
            )

        stage = "model_readiness"
        if readiness_check is not None:
            readiness_check(graph, dataset)

        stage = "behavioral_probes"
        sample_size = min(1024, len(dataset.X_train))
        assert_split_invariant(
            graph,
            _frame(dataset.X_train[:sample_size], graph.raw_columns),
            np.asarray(dataset.y_train[:sample_size]),
        )
        if graph.target is not None:
            if dataset.task_type != tabular_data.REGRESSION:
                raise ValueError("target transforms are supported for regression only")
            assert_target_round_trip(
                graph.target, np.asarray(dataset.y_train[:sample_size])
            )

        stage = "model_fit"
        lease_context = random_gpu_lease(gpu_model) if gpu_model else nullcontext(None)
        device_artifact: dict[str, Any] = {}
        model_failure: dict[str, object] | None = None
        with lease_context as lease:
            device = None if lease is None else lease.device
            if device_artifact_builder is not None:
                device_artifact = device_artifact_builder(device)
            try:

                def factory():
                    return model_builder(graph, device)

                metrics, evaluation_artifact = build(graph.dataset).validate(factory)
            except Exception as exc:
                model_failure = _failure_artifact(exc, stage)
                if exc.__traceback__ is not None:
                    traceback.clear_frames(exc.__traceback__)
                exc.__traceback__ = None
            finally:
                try:
                    if resource_cleanup is not None:
                        resource_cleanup()
                finally:
                    if lease is not None:
                        release_cuda(lease)
        if model_failure is not None:
            return dict(_INVALID), model_failure
        metrics.update(
            {
                "graph_node_count": float(len(graph.nodes)),
                "graph_max_depth": float(graph.depth),
                "generated_feature_count": float(len(graph.feature_output_columns)),
            }
        )
        artifact: dict[str, Any] = {
            "dataset": graph.dataset,
            "output_columns": graph.output_columns,
            "graph_node_count": len(graph.nodes),
            "estimator": estimator_name,
            "estimator_config": _config_payload(config),
        }
        if artifact_extra:
            artifact.update(artifact_extra)
        artifact.update(device_artifact)
        if isinstance(evaluation_artifact, dict):
            artifact.update(evaluation_artifact)
        return metrics, artifact
    except Exception as exc:
        return dict(_INVALID), _failure_artifact(exc, stage)


def score_payload_on_test(
    payload: object,
    *,
    model_builder: ModelBuilder,
    gpu_model: str | None = None,
    readiness_check: ReadinessCheck | None = None,
    resource_cleanup: ResourceCleanup | None = None,
) -> dict[str, float]:
    """Score a frozen graph on the untouched split under the same protocol."""

    graph = FeatureGraph.model_validate(payload)
    if readiness_check is not None:
        readiness_check(graph, tabular_data.load_dataset(graph.dataset))
    lease_context = random_gpu_lease(gpu_model) if gpu_model else nullcontext(None)
    failure: str | None = None
    result: dict[str, float] | None = None
    with lease_context as lease:
        device = None if lease is None else lease.device
        try:
            result = build(graph.dataset).score_on_test(
                lambda: model_builder(graph, device)
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
            exc.__traceback__ = None
        finally:
            try:
                if resource_cleanup is not None:
                    resource_cleanup()
            finally:
                if lease is not None:
                    release_cuda(lease)
    if failure is not None:
        raise RuntimeError(failure) from None
    if result is None:
        raise RuntimeError("test scorer returned no result")
    return result
