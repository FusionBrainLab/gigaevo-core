"""TabPFN v3 adapter for the canonical FeatureGraph execution contract."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

from problems.tabular_dag_baselines.foundation_backend import (
    InContextFeatureGraphModel,
)

_PREFIX = "GIGAEVO_TABPFN_"
_HF_REPO = "Prior-Labs/tabpfn_3"
_V3_CLASSIFIER_CHECKPOINT = "tabpfn-v3-classifier-v3_default.ckpt"
_V3_REGRESSOR_CHECKPOINT = "tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt"
TabPFNModelKind = Literal["classifier", "regressor"]


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.environ.get(_PREFIX + name, str(default)))
    if value < minimum:
        raise ValueError(f"{_PREFIX + name} must be >= {minimum}; got {value}")
    return value


@dataclass(frozen=True)
class TabPFNConfig:
    model_version: str = "v3"
    n_estimators: int = 8
    seed: int = 0
    fit_mode: str = "fit_preprocessors"
    model_path: str | None = None
    classifier_checkpoint: str = _V3_CLASSIFIER_CHECKPOINT
    # California's final context has 16,512 rows. The official medium-data v3
    # regressor is used rather than disabling TabPFN's pretraining limits.
    regressor_checkpoint: str = _V3_REGRESSOR_CHECKPOINT

    @classmethod
    def from_env(cls) -> TabPFNConfig:
        version = os.environ.get(_PREFIX + "MODEL_VERSION", "v3").lower()
        if version != "v3":
            raise ValueError(f"{_PREFIX}MODEL_VERSION must be 'v3'; got {version!r}")
        return cls(
            model_version=version,
            n_estimators=_env_int("N_ESTIMATORS", 8),
            seed=_env_int("SEED", 0, minimum=0),
            fit_mode=os.environ.get(_PREFIX + "FIT_MODE", "fit_preprocessors"),
            model_path=os.environ.get(_PREFIX + "MODEL_PATH"),
            classifier_checkpoint=os.environ.get(
                _PREFIX + "CLASSIFIER_CHECKPOINT", _V3_CLASSIFIER_CHECKPOINT
            ),
            regressor_checkpoint=os.environ.get(
                _PREFIX + "REGRESSOR_CHECKPOINT", _V3_REGRESSOR_CHECKPOINT
            ),
        )

    def checkpoint_name(self, which: TabPFNModelKind) -> str:
        return (
            self.regressor_checkpoint
            if which == "regressor"
            else self.classifier_checkpoint
        )


def ensure_tabpfn_ready(config: TabPFNConfig, *, which: TabPFNModelKind) -> Path:
    """Resolve/download the exact task-specific v3 checkpoint before GPU use."""

    if config.model_path is not None:
        path = Path(config.model_path).expanduser()
        if not path.is_file():
            raise RuntimeError(f"GIGAEVO_TABPFN_MODEL_PATH is not a file: {path}")
        if "v3" not in path.name.lower():
            raise RuntimeError(
                "GIGAEVO_TABPFN_MODEL_PATH must use a v3 checkpoint filename; "
                f"got {path.name!r}"
            )
        return path

    from tabpfn.constants import ModelVersion
    from tabpfn.model_loading import download_model, get_cache_dir

    checkpoint_name = config.checkpoint_name(which)
    path = get_cache_dir() / checkpoint_name
    if path.is_file():
        return path

    # The v3 repository exposes the released checkpoints through Hugging Face.
    # Download the exact selected filename directly so research runs do not
    # depend on the package's separate browser-login service.
    hf_error: Exception | None = None
    try:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo_id=_HF_REPO,
                filename=checkpoint_name,
                local_dir=get_cache_dir(),
            )
        )
        if downloaded.is_file():
            return downloaded
    except Exception as exc:
        hf_error = exc

    # Retain TabPFN's official PriorLabs authentication/download path as a
    # fallback for installations where direct Hugging Face access is gated.
    try:
        result = download_model(
            path,
            version=ModelVersion.V3,
            which=which,
            model_name=checkpoint_name,
        )
    except Exception as exc:
        result = [exc]
    if result != "ok" or not path.is_file():
        errors = result if isinstance(result, list) else []
        if hf_error is not None:
            errors.insert(0, hf_error)
        details = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
        raise RuntimeError(
            f"Could not prepare TabPFN v3 {which} checkpoint {checkpoint_name!r}: "
            f"{details or 'download returned without creating the checkpoint'}"
        )
    return path


class TabPFNFeatureGraphModel(InContextFeatureGraphModel):
    """FeatureGraph model using the package-pinned TabPFN v3 checkpoint."""

    estimator_name = "TabPFN-v3"
    gpu_model_name = "tabpfn"

    def __init__(self, graph, *, device: str | None = None, config=None):
        super().__init__(
            graph,
            device=device,
            config=config or TabPFNConfig.from_env(),
        )

    def _make_estimator(self, categorical_indices: list[int]):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import get_cache_dir

        common = {
            "n_estimators": self.config.n_estimators,
            "auto_scale_n_estimators": False,
            "categorical_features_indices": categorical_indices,
            "device": self.device,
            "fit_mode": self.config.fit_mode,
            "random_state": self.config.seed,
            "show_progress_bar": False,
        }
        model_class = (
            TabPFNRegressor if self.task_type == "regression" else TabPFNClassifier
        )
        which: TabPFNModelKind = (
            "regressor" if self.task_type == "regression" else "classifier"
        )
        if self.config.model_path is not None:
            model_path = Path(self.config.model_path).expanduser()
        else:
            model_path = get_cache_dir() / self.config.checkpoint_name(which)
        common["model_path"] = str(model_path)
        return model_class.create_default_for_version(ModelVersion.V3, **common)


FeatureGraphModel = TabPFNFeatureGraphModel
