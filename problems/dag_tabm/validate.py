"""GigaEvo validator for FeatureGraph genomes scored by TabM."""

from __future__ import annotations

from dataclasses import asdict
import gc
from pathlib import Path
import sys

import numpy as np

_SOURCE_PATH = globals().get("__file__")
_PROBLEM_DIR = (
    Path(_SOURCE_PATH).resolve().parent if _SOURCE_PATH else Path(sys.path[0]).resolve()
)
_TABULAR_COMMON = _PROBLEM_DIR.parent / "tabular" / "_common"
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
from problems.dag_tabm.gpu_pool import random_gpu_lease  # noqa: E402
from problems.dag_tabm.tabm_backend import (  # noqa: E402
    FeatureGraphModel,
    TabMConfig,
    effective_amp_dtype,
)


def _factory(graph: FeatureGraph, device: str, config: TabMConfig):
    return lambda: FeatureGraphModel(graph, device=device, config=config)


def _release_cuda(lease) -> None:
    if lease.logical_index is None:
        return
    import torch

    gc.collect()
    with torch.cuda.device(lease.logical_index):
        torch.cuda.empty_cache()


def validate(payload):
    """Validate a FeatureGraph and score it with fixed TabM-mini+PLE."""

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
        config = TabMConfig.from_env()
        with random_gpu_lease() as lease:
            try:
                amp_dtype = effective_amp_dtype(lease.device, enabled=config.amp)
                metrics, evaluation_artifact = build(graph.dataset).validate(
                    _factory(graph, lease.device, config)
                )
            finally:
                _release_cuda(lease)
        metrics.update(
            {
                "graph_node_count": float(len(graph.nodes)),
                "graph_max_depth": float(graph.depth),
                "generated_feature_count": float(len(graph.feature_output_columns)),
            }
        )
        artifact = {
            "dataset": graph.dataset,
            "output_columns": graph.output_columns,
            "graph_node_count": len(graph.nodes),
            "estimator": f"{config.arch_type}-ple",
            "tabm_k": config.k,
            "tabm_refit": config.refit,
            "tabm_share_training_batches": config.share_training_batches,
            "tabm_amp_dtype": amp_dtype,
            "tabm_config": asdict(config),
        }
        if isinstance(evaluation_artifact, dict):
            artifact.update(evaluation_artifact)
        return metrics, artifact
    except Exception as exc:
        return dict(_INVALID), _failure_artifact(exc, stage)


def score_on_test(payload):
    """Score on the untouched test split with the shared tabular protocol."""

    graph = FeatureGraph.model_validate(payload)
    config = TabMConfig.from_env()
    with random_gpu_lease() as lease:
        try:
            return build(graph.dataset).score_on_test(
                _factory(graph, lease.device, config)
            )
        finally:
            _release_cuda(lease)
