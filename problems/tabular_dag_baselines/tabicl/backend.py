"""TabICLv2 adapter for the canonical FeatureGraph execution contract."""

from __future__ import annotations

from dataclasses import dataclass
import os

from problems.tabular_dag_baselines.foundation_backend import (
    InContextFeatureGraphModel,
)

_PREFIX = "GIGAEVO_TABICL_"


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.environ.get(_PREFIX + name, str(default)))
    if value < minimum:
        raise ValueError(f"{_PREFIX + name} must be >= {minimum}; got {value}")
    return value


@dataclass(frozen=True)
class TabICLConfig:
    n_estimators: int = 8
    batch_size: int = 8
    seed: int = 0
    classifier_checkpoint: str = "tabicl-classifier-v2-20260212.ckpt"
    regressor_checkpoint: str = "tabicl-regressor-v2-20260212.ckpt"
    model_path: str | None = None

    @classmethod
    def from_env(cls) -> TabICLConfig:
        return cls(
            n_estimators=_env_int("N_ESTIMATORS", 8),
            batch_size=_env_int("BATCH_SIZE", 8),
            seed=_env_int("SEED", 0, minimum=0),
            classifier_checkpoint=os.environ.get(
                _PREFIX + "CLASSIFIER_CHECKPOINT",
                "tabicl-classifier-v2-20260212.ckpt",
            ),
            regressor_checkpoint=os.environ.get(
                _PREFIX + "REGRESSOR_CHECKPOINT",
                "tabicl-regressor-v2-20260212.ckpt",
            ),
            model_path=os.environ.get(_PREFIX + "MODEL_PATH"),
        )


class TabICLFeatureGraphModel(InContextFeatureGraphModel):
    """FeatureGraph model using the frozen February 2026 TabICLv2 checkpoint."""

    estimator_name = "TabICLv2"
    gpu_model_name = "tabicl"

    def __init__(self, graph, *, device: str | None = None, config=None):
        super().__init__(
            graph,
            device=device,
            config=config or TabICLConfig.from_env(),
        )

    def _make_estimator(self, categorical_indices: list[int]):
        from tabicl import TabICLClassifier, TabICLRegressor

        del categorical_indices  # TabICL detects pandas categorical dtypes.
        common = {
            "n_estimators": self.config.n_estimators,
            "batch_size": self.config.batch_size,
            "model_path": self.config.model_path,
            "device": self.device,
            "random_state": self.config.seed,
            "verbose": False,
        }
        if self.task_type == "regression":
            return TabICLRegressor(
                checkpoint_version=self.config.regressor_checkpoint,
                **common,
            )
        return TabICLClassifier(
            checkpoint_version=self.config.classifier_checkpoint,
            **common,
        )


FeatureGraphModel = TabICLFeatureGraphModel
