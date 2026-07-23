"""TabM estimator adapter for the existing FeatureGraph execution contract."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from problems.dag_tab.execution import (
    GraphTriplet,
    assert_target_round_trip,
    inverse_target,
    transform_target,
)
from problems.dag_tab.validate import FeatureGraphModel as _GraphFeatureModel
from problems.tabular_dag_baselines.gpu_pool import random_gpu_lease, release_cuda

_PREFIX = "GIGAEVO_TABM_"


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.environ.get(_PREFIX + name, str(default)))
    if value < minimum:
        raise ValueError(f"{_PREFIX + name} must be >= {minimum}; got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    value = float(os.environ.get(_PREFIX + name, str(default)))
    if not math.isfinite(value) or value < minimum:
        raise ValueError(
            f"{_PREFIX + name} must be finite and >= {minimum}; got {value}"
        )
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(_PREFIX + name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{_PREFIX + name} must be a boolean; got {raw!r}")


@dataclass(frozen=True)
class TabMConfig:
    """Fixed California configuration, with explicit screening overrides."""

    arch_type: str = "tabm-mini"
    k: int = 32
    n_blocks: int = 3
    d_block: int = 576
    dropout: float = 0.24050495351031098
    learning_rate: float = 0.00029926241255995084
    weight_decay: float = 0.0
    n_bins: int = 30
    d_embedding: int = 16
    batch_size: int = 256
    eval_batch_size: int = 8192
    patience: int = 16
    max_epochs: int = 512
    gradient_clipping_norm: float = 1.0
    seed: int = 0
    amp: bool = True
    share_training_batches: bool = False
    refit: bool = True

    @classmethod
    def from_env(cls) -> TabMConfig:
        arch_type = os.environ.get(_PREFIX + "ARCH_TYPE", "tabm-mini")
        if arch_type not in {"tabm", "tabm-mini"}:
            raise ValueError(
                f"{_PREFIX}ARCH_TYPE must be 'tabm' or 'tabm-mini'; got {arch_type!r}"
            )
        return cls(
            arch_type=arch_type,
            k=_env_int("K", 32),
            n_blocks=_env_int("N_BLOCKS", 3),
            d_block=_env_int("D_BLOCK", 576),
            dropout=_env_float("DROPOUT", 0.24050495351031098),
            learning_rate=_env_float("LEARNING_RATE", 0.00029926241255995084),
            weight_decay=_env_float("WEIGHT_DECAY", 0.0),
            n_bins=_env_int("N_BINS", 30, minimum=2),
            d_embedding=_env_int("D_EMBEDDING", 16),
            batch_size=_env_int("BATCH_SIZE", 256),
            eval_batch_size=_env_int("EVAL_BATCH_SIZE", 8192),
            patience=_env_int("PATIENCE", 16, minimum=0),
            max_epochs=_env_int("MAX_EPOCHS", 512),
            gradient_clipping_norm=_env_float("GRADIENT_CLIPPING_NORM", 1.0),
            seed=_env_int("SEED", 0, minimum=0),
            amp=_env_bool("AMP", True),
            share_training_batches=_env_bool("SHARE_TRAINING_BATCHES", False),
            refit=_env_bool("REFIT", True),
        )


@dataclass
class _PreparedFeatures:
    x_num: dict[str, np.ndarray | None]
    x_cat: dict[str, np.ndarray | None]
    cat_cardinalities: list[int]

    @property
    def n_num_features(self) -> int:
        values = self.x_num["fit"]
        return 0 if values is None else int(values.shape[1])


@dataclass
class _TargetScale:
    mean: float
    std: float
    reference_y: np.ndarray


@dataclass
class _TrainingResult:
    model: Any
    best_epochs: int


def _category_key(value: object) -> tuple[str, str, str]:
    if pd.isna(value):
        return ("missing", "", "")
    return ("value", type(value).__qualname__, str(value))


def _weighted_mean(values, weights):
    if weights is None:
        return values.mean()
    denominator = weights.sum().clamp_min(1e-12)
    return (values * weights).sum() / denominator


def _amp_dtype(torch, device, enabled: bool):
    """Use the paper-tested BF16 path where available, with FP16 fallback."""

    if not enabled or device.type != "cuda":
        return None
    with torch.cuda.device(device):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return torch.float16


def effective_amp_dtype(device_name: str, *, enabled: bool) -> str | None:
    """Return the effective autocast dtype for provenance artifacts."""

    import torch

    dtype = _amp_dtype(torch, torch.device(device_name), enabled)
    return None if dtype is None else str(dtype).removeprefix("torch.")


def _training_batches(torch, train_size, config, device, generator):
    """Draw batches from a dedicated RNG, independent of model initialization."""

    if config.share_training_batches:
        return torch.randperm(train_size, generator=generator, device=device).split(
            config.batch_size
        )
    return (
        torch.rand((train_size, config.k), generator=generator, device=device)
        .argsort(dim=0)
        .split(config.batch_size, dim=0)
    )


class TabMFeatureGraphModel(_GraphFeatureModel):
    """FeatureGraph model using the tuned California TabM-mini+PLE recipe."""

    def __init__(
        self, graph, *, device: str | None = None, config: TabMConfig | None = None
    ):
        super().__init__(graph)
        self.device = device
        self.config = config or TabMConfig.from_env()
        self.last_fit_summary: dict[str, Any] = {}

    def _prepare_features(self, triplet: GraphTriplet) -> _PreparedFeatures:
        frames = {
            "fit": triplet.fit,
            "validation": triplet.validation,
            "query": triplet.query,
        }
        expected_columns = list(triplet.fit.columns)
        for name, frame in frames.items():
            if list(frame.columns) != expected_columns:
                raise ValueError(f"{name} feature columns differ from fit columns")

        num_columns = [
            column
            for column in expected_columns
            if self._feature_kind(column) not in {"categorical", "binary"}
        ]
        cat_columns = [
            column for column in expected_columns if column not in num_columns
        ]

        x_num: dict[str, np.ndarray | None] = {name: None for name in frames}
        if num_columns:
            numeric: dict[str, np.ndarray] = {}
            for name, frame in frames.items():
                columns = [
                    pd.to_numeric(frame[column], errors="raise").to_numpy(
                        dtype=np.float64
                    )
                    for column in num_columns
                ]
                numeric[name] = np.column_stack(columns)
                if np.isinf(numeric[name]).any():
                    raise ValueError(f"{name} numerical features contain infinity")

            fit_values = numeric["fit"]
            medians = np.asarray(
                [
                    np.nanmedian(column) if np.isfinite(column).any() else 0.0
                    for column in fit_values.T
                ],
                dtype=np.float64,
            )
            for name, values in numeric.items():
                missing_rows, missing_columns = np.where(np.isnan(values))
                if len(missing_rows):
                    values = values.copy()
                    values[missing_rows, missing_columns] = medians[missing_columns]
                numeric[name] = values

            # Piecewise-linear bins require non-constant training columns.  This
            # is the same filtering used by the paper preprocessing pipeline.
            keep = np.ptp(numeric["fit"], axis=0) > 0
            if np.any(keep):
                for name in numeric:
                    numeric[name] = numeric[name][:, keep]
                noise = np.random.RandomState(self.config.seed).normal(
                    0.0, 1e-5, numeric["fit"].shape
                )
                transformer = QuantileTransformer(
                    n_quantiles=min(
                        len(numeric["fit"]),
                        max(min(len(numeric["fit"]) // 30, 1000), 10),
                    ),
                    output_distribution="normal",
                    subsample=1_000_000_000,
                    random_state=self.config.seed,
                ).fit(numeric["fit"] + noise)
                for name, values in numeric.items():
                    transformed = (
                        transformer.transform(values)
                        if len(values)
                        else np.empty((0, int(keep.sum())), dtype=np.float64)
                    )
                    x_num[name] = np.nan_to_num(transformed).astype(np.float32)

        x_cat: dict[str, np.ndarray | None] = {name: None for name in frames}
        cardinalities: list[int] = []
        if cat_columns:
            encoded = {
                name: np.empty((len(frame), len(cat_columns)), dtype=np.int64)
                for name, frame in frames.items()
            }
            for column_index, column in enumerate(cat_columns):
                keys = [
                    _category_key(value) for value in frames["fit"][column].tolist()
                ]
                mapping = {key: index for index, key in enumerate(dict.fromkeys(keys))}
                unknown = len(mapping)
                cardinalities.append(unknown + 1)
                for name, frame in frames.items():
                    encoded[name][:, column_index] = [
                        mapping.get(_category_key(value), unknown)
                        for value in frame[column].tolist()
                    ]
            x_cat.update(encoded)

        if x_num["fit"] is None and x_cat["fit"] is None:
            raise ValueError("FeatureGraph produced no usable estimator features")
        return _PreparedFeatures(x_num, x_cat, cardinalities)

    def _prepare_targets(
        self,
        fit_y: np.ndarray,
        validation_y: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, _TargetScale]:
        if self.task_type == "regression":
            transformed_fit = np.asarray(
                transform_target(self.graph.target, fit_y, fit_y), dtype=np.float32
            )
            transformed_validation = (
                None
                if validation_y is None
                else np.asarray(
                    transform_target(self.graph.target, fit_y, validation_y),
                    dtype=np.float32,
                )
            )
            mean = float(transformed_fit.mean())
            std = float(transformed_fit.std())
            if not math.isfinite(std) or std < 1e-12:
                std = 1.0
            return (
                (transformed_fit - mean) / std,
                None
                if transformed_validation is None
                else (transformed_validation - mean) / std,
                _TargetScale(mean, std, fit_y.copy()),
            )

        return (
            fit_y.astype(np.int64),
            None if validation_y is None else validation_y.astype(np.int64),
            _TargetScale(0.0, 1.0, fit_y.copy()),
        )

    def _validate_classification_labels(
        self, fit_y: np.ndarray, validation_y: np.ndarray
    ) -> None:
        if self.n_classes is None or self.n_classes < 2:
            raise ValueError("classification dataset must declare n_classes >= 2")
        observed = np.unique(np.concatenate([fit_y, validation_y]).astype(int))
        if np.any(observed < 0) or np.any(observed >= int(self.n_classes)):
            raise ValueError(
                f"classification labels {observed.tolist()} fall outside declared "
                f"class universe [0, {self.n_classes})"
            )

    def _fit_model(
        self,
        features: _PreparedFeatures,
        fit_y: np.ndarray,
        validation_y: np.ndarray | None,
        fit_weight: np.ndarray | None,
        validation_weight: np.ndarray | None,
        *,
        device_name: str,
        fixed_epochs: int | None = None,
    ) -> _TrainingResult:
        import rtdl_num_embeddings
        import tabm
        import torch
        import torch.nn.functional as functional

        config = self.config
        device = torch.device(device_name)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        def tensor(value, *, dtype=None):
            return (
                None
                if value is None
                else torch.as_tensor(value, dtype=dtype, device=device)
            )

        x_num = {
            name: tensor(value, dtype=torch.float32)
            for name, value in features.x_num.items()
        }
        x_cat = {
            name: tensor(value, dtype=torch.long)
            for name, value in features.x_cat.items()
        }
        regression = self.task_type == "regression"
        target_dtype = torch.float32 if regression else torch.long
        targets = {
            "fit": tensor(fit_y, dtype=target_dtype),
            "validation": tensor(validation_y, dtype=target_dtype),
        }
        weights = {
            "fit": tensor(fit_weight, dtype=torch.float32),
            "validation": tensor(validation_weight, dtype=torch.float32),
        }

        num_embeddings = None
        if features.n_num_features:
            bins = rtdl_num_embeddings.compute_bins(x_num["fit"], n_bins=config.n_bins)
            num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                bins,
                d_embedding=config.d_embedding,
                activation=False,
                version="B",
            )

        model = tabm.TabM.make(
            n_num_features=features.n_num_features,
            cat_cardinalities=features.cat_cardinalities,
            d_out=1 if regression else int(self.n_classes),
            num_embeddings=num_embeddings,
            arch_type=config.arch_type,
            k=config.k,
            n_blocks=config.n_blocks,
            d_block=config.d_block,
            dropout=config.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        # Keep the training-object order fixed across feature graphs. Model
        # initialization consumes a graph-dependent number of random values, so
        # the global RNG would otherwise give different batch permutations to
        # candidates with different input widths.
        batch_generator = torch.Generator(device=device).manual_seed(config.seed)
        amp_dtype = _amp_dtype(torch, device, config.amp)
        amp_enabled = amp_dtype is not None
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

        def apply(part: str, indices):
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=amp_dtype or torch.float16,
            ):
                return model(
                    None if x_num[part] is None else x_num[part][indices],
                    None if x_cat[part] is None else x_cat[part][indices],
                )

        def validation_loss():
            assert targets["validation"] is not None
            model.eval()
            predictions = []
            with torch.inference_mode():
                all_indices = torch.arange(
                    len(targets["validation"]), device=device
                ).split(config.eval_batch_size)
                for indices in all_indices:
                    predictions.append(apply("validation", indices).float())
            prediction = torch.cat(predictions)
            if regression:
                prediction = prediction.squeeze(-1).mean(1)
                losses = functional.mse_loss(
                    prediction, targets["validation"], reduction="none"
                )
            else:
                probability = prediction.softmax(-1).mean(1).clamp_min(1e-12)
                losses = (
                    -probability.gather(1, targets["validation"].unsqueeze(1))
                    .squeeze(1)
                    .log()
                )
            return float(_weighted_mean(losses, weights["validation"]))

        train_size = len(fit_y)
        max_epochs = fixed_epochs if fixed_epochs is not None else config.max_epochs
        best_loss = math.inf
        best_epoch = -1
        best_state = None
        remaining_patience = config.patience
        stop_epoch = max_epochs - 1

        for epoch in range(max_epochs):
            model.train()
            batches = _training_batches(
                torch,
                train_size,
                config,
                device,
                batch_generator,
            )
            for indices in batches:
                optimizer.zero_grad(set_to_none=True)
                prediction = apply("fit", indices)
                if config.share_training_batches:
                    target = targets["fit"][indices].unsqueeze(1).expand(-1, config.k)
                    batch_weight = (
                        None
                        if weights["fit"] is None
                        else weights["fit"][indices].unsqueeze(1).expand(-1, config.k)
                    )
                else:
                    target = targets["fit"][indices]
                    batch_weight = (
                        None if weights["fit"] is None else weights["fit"][indices]
                    )

                if regression:
                    losses = functional.mse_loss(
                        prediction.squeeze(-1), target, reduction="none"
                    )
                else:
                    losses = functional.cross_entropy(
                        prediction.flatten(0, 1),
                        target.flatten(),
                        reduction="none",
                    ).reshape_as(target)
                loss = _weighted_mean(losses, batch_weight)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clipping_norm
                )
                scaler.step(optimizer)
                scaler.update()

            if fixed_epochs is None:
                score = validation_loss()
                if score < best_loss:
                    best_loss = score
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().clone()
                        for name, value in model.state_dict().items()
                    }
                    remaining_patience = config.patience
                else:
                    remaining_patience -= 1
                if remaining_patience < 0:
                    stop_epoch = epoch
                    break

        if fixed_epochs is None:
            if best_state is None:
                raise RuntimeError("TabM training did not produce a finite checkpoint")
            model.load_state_dict(best_state)
            best_epochs = best_epoch + 1
        else:
            best_epochs = fixed_epochs

        self.last_fit_summary = {
            "best_epochs": best_epochs,
            "stop_epoch": stop_epoch,
            "best_validation_loss": None if fixed_epochs is not None else best_loss,
            "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "amp_dtype": None
            if amp_dtype is None
            else str(amp_dtype).removeprefix("torch."),
        }
        return _TrainingResult(model, best_epochs)

    def _predict(
        self,
        model,
        features: _PreparedFeatures,
        scale: _TargetScale,
        *,
        device_name: str,
    ) -> np.ndarray:
        import torch

        device = torch.device(device_name)
        config = self.config
        x_num = (
            None
            if features.x_num["query"] is None
            else torch.as_tensor(
                features.x_num["query"], dtype=torch.float32, device=device
            )
        )
        x_cat = (
            None
            if features.x_cat["query"] is None
            else torch.as_tensor(
                features.x_cat["query"], dtype=torch.long, device=device
            )
        )
        outputs = []
        model.eval()
        amp_dtype = _amp_dtype(torch, device, config.amp)
        with torch.inference_mode():
            for indices in torch.arange(
                len(features.x_num["query"])
                if features.x_num["query"] is not None
                else len(features.x_cat["query"]),
                device=device,
            ).split(config.eval_batch_size):
                with torch.autocast(
                    device_type=device.type,
                    enabled=amp_dtype is not None,
                    dtype=amp_dtype or torch.float16,
                ):
                    outputs.append(
                        model(
                            None if x_num is None else x_num[indices],
                            None if x_cat is None else x_cat[indices],
                        ).float()
                    )
        prediction = torch.cat(outputs)
        if self.task_type == "regression":
            values = prediction.squeeze(-1).mean(1).cpu().numpy()
            transformed = values * scale.std + scale.mean
            return np.asarray(
                inverse_target(self.graph.target, scale.reference_y, transformed),
                dtype=float,
            )
        return prediction.softmax(-1).mean(1).cpu().numpy()

    def _fit_predict_on_device(
        self, X_train, y_train, X_val, y_val, X_query, *, device_name: str
    ) -> np.ndarray:
        train_y = np.asarray(y_train)
        val_y = np.asarray(y_val)
        if self.graph.target is not None and self.task_type != "regression":
            raise ValueError("target transforms are supported for regression only")
        if self.graph.target is not None:
            assert_target_round_trip(self.graph.target, train_y)
        if self.task_type != "regression":
            self._validate_classification_labels(train_y, val_y)

        search_triplet, train_weight, val_weight = self._extract_sample_weights(
            self._transform(X_train, train_y, X_val, X_query)
        )
        search_features = self._prepare_features(search_triplet)
        search_train_y, search_val_y, search_scale = self._prepare_targets(
            train_y, val_y
        )
        search_result = self._fit_model(
            search_features,
            search_train_y,
            search_val_y,
            train_weight,
            val_weight,
            device_name=device_name,
        )
        if not self.config.refit:
            return self._predict(
                search_result.model,
                search_features,
                search_scale,
                device_name=device_name,
            )

        del search_result.model
        gc.collect()

        fit_X = np.concatenate([np.asarray(X_train), np.asarray(X_val)])
        fit_y = np.concatenate([train_y, val_y])
        empty = np.asarray(X_val)[:0]
        final_triplet, fit_weight, _ = self._extract_sample_weights(
            self._transform(fit_X, fit_y, empty, X_query)
        )
        final_features = self._prepare_features(final_triplet)
        final_y, _, final_scale = self._prepare_targets(fit_y, None)
        final_result = self._fit_model(
            final_features,
            final_y,
            None,
            fit_weight,
            None,
            device_name=device_name,
            fixed_epochs=search_result.best_epochs,
        )
        return self._predict(
            final_result.model,
            final_features,
            final_scale,
            device_name=device_name,
        )

    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        if self.device is not None:
            return self._fit_predict_on_device(
                X_train, y_train, X_val, y_val, X_query, device_name=self.device
            )

        with random_gpu_lease("tabm") as lease:
            try:
                return self._fit_predict_on_device(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_query,
                    device_name=lease.device,
                )
            finally:
                release_cuda(lease)


# Keep the conventional name used by the other tabular adapters.
FeatureGraphModel = TabMFeatureGraphModel
