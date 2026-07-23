"""TabFM 1.0 PyTorch adapter for the canonical FeatureGraph contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Literal

from problems.tabular_dag_baselines.foundation_backend import (
    InContextFeatureGraphModel,
)

_PREFIX = "GIGAEVO_TABFM_"
_HF_REPO = "google/tabfm-1.0.0-pytorch"
_HF_REVISION = "77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83"
_CHECKPOINT_FILES = ("config.json", "model.safetensors")
TABFM_MAX_CLASSES = 10
TabFMModelType = Literal["classification", "regression"]
ModelCache = dict[tuple[TabFMModelType, str | None, str], Any]


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.environ.get(_PREFIX + name, str(default)))
    if value < minimum:
        raise ValueError(f"{_PREFIX + name} must be >= {minimum}; got {value}")
    return value


@dataclass(frozen=True)
class TabFMConfig:
    model_release: str = "1.0.0"
    checkpoint_repo: str = _HF_REPO
    checkpoint_revision: str = _HF_REVISION
    n_estimators: int = 32
    batch_size: int = 1
    max_num_features: int = 500
    max_num_rows: int | None = None
    seed: int = 0
    model_path: str | None = None

    @classmethod
    def from_env(cls) -> TabFMConfig:
        return cls(
            n_estimators=_env_int("N_ESTIMATORS", 32),
            batch_size=_env_int("BATCH_SIZE", 1),
            max_num_features=_env_int("MAX_NUM_FEATURES", 500),
            max_num_rows=(
                _env_int("MAX_NUM_ROWS", 1)
                if _PREFIX + "MAX_NUM_ROWS" in os.environ
                else None
            ),
            seed=_env_int("SEED", 0, minimum=0),
            model_path=os.environ.get(_PREFIX + "MODEL_PATH"),
        )


def _validate_checkpoint(
    path: Path, *, model_type: TabFMModelType, setting: str
) -> Path:
    selected = path / model_type if (path / model_type).is_dir() else path
    if not selected.is_dir():
        raise RuntimeError(f"{setting} is not a directory: {path}")
    missing = [name for name in _CHECKPOINT_FILES if not (selected / name).is_file()]
    if missing:
        raise RuntimeError(
            f"{setting} does not contain a complete TabFM {model_type} checkpoint "
            f"under {selected}: missing {', '.join(missing)}"
        )
    try:
        metadata = json.loads((selected / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"{setting} has an invalid config.json under {selected}"
        ) from exc
    expected_classifier = model_type == "classification"
    if metadata.get("is_classifier") is not expected_classifier:
        raise RuntimeError(
            f"{setting} checkpoint task does not match requested {model_type}: "
            f"{selected}"
        )
    return path


def ensure_tabfm_ready(config: TabFMConfig, *, model_type: TabFMModelType) -> Path:
    """Resolve the pinned task checkpoint before taking a GPU lease."""

    if config.model_path is not None:
        path = Path(config.model_path).expanduser().resolve()
        return _validate_checkpoint(
            path, model_type=model_type, setting="GIGAEVO_TABFM_MODEL_PATH"
        )

    from huggingface_hub import snapshot_download

    download_args: dict[str, Any] = {
        "repo_id": config.checkpoint_repo,
        "revision": config.checkpoint_revision,
        "allow_patterns": [
            f"{model_type}/config.json",
            f"{model_type}/model.safetensors",
        ],
    }
    try:
        resolved = snapshot_download(**download_args, local_files_only=True)
        return _validate_checkpoint(
            Path(resolved), model_type=model_type, setting="cached checkpoint"
        )
    except Exception:
        try:
            resolved = snapshot_download(**download_args)
            return _validate_checkpoint(
                Path(resolved),
                model_type=model_type,
                setting="downloaded checkpoint",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not prepare pinned TabFM {model_type} checkpoint: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def _load_tabfm_model(
    *,
    model_type: TabFMModelType,
    checkpoint_path: Path,
    device: str | None,
):
    from tabfm import tabfm_v1_0_0_pytorch

    # The adapter owns an evaluation-scoped cache. Avoid TabFM's process-wide
    # cache, which would retain a 6+ GB model after the GPU lease is released.
    return tabfm_v1_0_0_pytorch.load(
        model_type=model_type,
        checkpoint_path=str(checkpoint_path),
        device=device,
        use_cache=False,
    )


class TabFMFeatureGraphModel(InContextFeatureGraphModel):
    """FeatureGraph model using the frozen TabFM 1.0 PyTorch checkpoint."""

    estimator_name = "TabFM-1.0"
    gpu_model_name = "tabfm"

    def __init__(
        self,
        graph,
        *,
        device: str | None = None,
        config=None,
        model_cache: ModelCache | None = None,
    ):
        super().__init__(
            graph,
            device=device,
            config=config or TabFMConfig.from_env(),
        )
        self.model_cache = model_cache if model_cache is not None else {}

    def _make_estimator(self, categorical_indices: list[int]):
        from tabfm import TabFMClassifier, TabFMRegressor

        del categorical_indices  # TabFM detects pandas categorical dtypes.
        model_type: TabFMModelType = (
            "regression" if self.task_type == "regression" else "classification"
        )
        checkpoint_identity = (
            self.config.model_path
            or f"{self.config.checkpoint_repo}@{self.config.checkpoint_revision}"
        )
        cache_key = (model_type, self.device, checkpoint_identity)
        if cache_key not in self.model_cache:
            checkpoint_path = ensure_tabfm_ready(self.config, model_type=model_type)
            self.model_cache[cache_key] = _load_tabfm_model(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                device=self.device,
            )

        common = {
            "model": self.model_cache[cache_key],
            "n_estimators": self.config.n_estimators,
            "batch_size": self.config.batch_size,
            "max_num_features": self.config.max_num_features,
            "max_num_rows": self.config.max_num_rows,
            "random_state": self.config.seed,
            "verbose": False,
        }
        if self.task_type == "regression":
            return TabFMRegressor(**common)
        return TabFMClassifier(**common)


FeatureGraphModel = TabFMFeatureGraphModel
