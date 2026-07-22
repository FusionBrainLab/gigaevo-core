from __future__ import annotations

import os

import numpy as np
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold
import tabular_bd
import tabular_data
import tabular_metrics

CV_FOLDS_ENV = "GIGAEVO_TABULAR_CV_FOLDS"
CV_MAX_ENV = "GIGAEVO_TABULAR_CV_MAX"
BD_MAX_ENV = "GIGAEVO_TABULAR_BD_MAX"
FITNESS_ENV = "GIGAEVO_TABULAR_FITNESS"
LCB_LAMBDA_ENV = "GIGAEVO_TABULAR_LCB_LAMBDA"

_ALLOWED_FOLDS = {2, 3, 5, 10}
_DEFAULT_FOLDS = 3
_DEFAULT_CV_MAX = 100_000
_DEFAULT_BD_MAX = 2048
_HOLDOUT_SPLITS = 5  # first fold of a 5-way split == 80/20 holdout
_SEED = 0
_DEFAULT_LCB_LAMBDA = 1.0  # lambda=1 == the textbook 1-standard-error rule
_EVALUATION_MEASUREMENTS_KEY = "_evaluation_measurements"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from e


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be a float; got {raw!r}") from e


def _fitness_mode() -> str:
    mode = os.environ.get(FITNESS_ENV, "mean").lower()
    if mode not in {"mean", "lcb"}:
        raise ValueError(f"{FITNESS_ENV} must be 'mean' or 'lcb'; got {mode!r}")
    return mode


def _aggregate_fitness(mean_score: float, cv_std: float, n_folds: int) -> float:
    mode = _fitness_mode()
    if mode == "mean":
        return mean_score
    if mode == "lcb":
        if n_folds <= 1:
            return mean_score
        lam = _env_float(LCB_LAMBDA_ENV, _DEFAULT_LCB_LAMBDA)
        return mean_score - lam * cv_std / np.sqrt(n_folds)
    raise AssertionError(f"unhandled fitness mode {mode!r}")


def _evaluation_measurement_artifact(
    cv_score_std: float, n_folds: int
) -> dict[str, object] | None:
    """Report uncertainty only for the mean-fitness CV estimator."""

    if n_folds <= 1 or _fitness_mode() != "mean":
        return None
    return {
        _EVALUATION_MEASUREMENTS_KEY: {
            "fitness": {
                "sample_sd": cv_score_std,
                "n": n_folds,
                "method": "cross_validation",
            }
        }
    }


def _k_folds() -> int:
    k = _env_int(CV_FOLDS_ENV, _DEFAULT_FOLDS)
    if k not in _ALLOWED_FOLDS:
        raise ValueError(
            f"{CV_FOLDS_ENV} must be one of {sorted(_ALLOWED_FOLDS)}; got {k}"
        )
    return k


class TabularProblem:
    def __init__(self, name: str):
        self.name = name

    def _dataset(self) -> tabular_data.Dataset:
        return tabular_data.load_dataset(self.name)

    def _splits(self, n: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return rotating train, early-stop validation, and query indices."""

        idx = np.arange(n)
        if n <= _env_int(CV_MAX_ENV, _DEFAULT_CV_MAX):
            kf = KFold(n_splits=_k_folds(), shuffle=True, random_state=_SEED)
            query_folds = [query_idx for _, query_idx in kf.split(idx)]
            query_positions = range(len(query_folds))
        else:
            kf = KFold(n_splits=_HOLDOUT_SPLITS, shuffle=True, random_state=_SEED)
            query_folds = [query_idx for _, query_idx in kf.split(idx)]
            query_positions = range(1)

        splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for query_position in query_positions:
            val_position = (query_position + 1) % len(query_folds)
            if len(query_folds) == 2:
                fit_pool = query_folds[val_position]
                inner = KFold(n_splits=2, shuffle=True, random_state=_SEED)
                train_local, val_local = next(inner.split(fit_pool))
                splits.append(
                    (
                        fit_pool[train_local],
                        fit_pool[val_local],
                        query_folds[query_position],
                    )
                )
                continue
            train_idx = np.concatenate(
                [
                    fold
                    for position, fold in enumerate(query_folds)
                    if position not in {query_position, val_position}
                ]
            )
            splits.append(
                (train_idx, query_folds[val_position], query_folds[query_position])
            )
        return splits

    def _fold_metrics(self, ds, y_true, pred) -> dict[str, float]:
        if ds.task_type == tabular_data.REGRESSION:
            return tabular_metrics.regression_fold_metrics(y_true, pred)
        return tabular_metrics.classification_fold_metrics(
            y_true, pred, ds.n_classes, ds.task_type
        )

    def _bd(self, ds, model_factory) -> dict[str, float]:
        sd_y = (
            float(np.std(ds.y_train) + tabular_bd.BD_EPS)
            if ds.task_type == tabular_data.REGRESSION
            else 1.0
        )
        try:
            return tabular_bd.compute_bd_axes(
                model_factory,
                ds.X_train.copy(),
                ds.y_train.copy(),
                ds.X_val.copy(),
                ds.y_val.copy(),
                task_type=ds.task_type,
                sd_y=sd_y,
                bd_max=_env_int(BD_MAX_ENV, _DEFAULT_BD_MAX),
            )
        except Exception:
            return {
                "local_lipschitz_p95": tabular_bd.BD_LL_MAX,
                "ood_delta_slope": tabular_bd.BD_OOD_MAX,
            }

    def validate(
        self, model_factory
    ) -> tuple[dict[str, float], dict[str, object] | None]:
        ds = self._dataset()
        X_dev = np.concatenate([ds.X_train, ds.X_val])
        y_dev = np.concatenate([ds.y_train, ds.y_val])
        splits = self._splits(X_dev.shape[0])
        folds: list[dict[str, float]] = []
        for train_idx, val_idx, query_idx in splits:
            instance = tabular_metrics.instantiate(model_factory)
            pred = instance.fit_predict(
                X_dev[train_idx],
                y_dev[train_idx],
                X_dev[val_idx],
                y_dev[val_idx],
                X_dev[query_idx],
            )
            folds.append(self._fold_metrics(ds, y_dev[query_idx], pred))

        scores = [m["score"] for m in folds]
        cv_score_std = float(np.std(scores, ddof=1)) if len(folds) > 1 else 0.0
        fitness = _aggregate_fitness(float(np.mean(scores)), cv_score_std, len(folds))
        secondary_keys = [k for k in folds[0] if k != "score"]
        secondary = {k: float(np.mean([m[k] for m in folds])) for k in secondary_keys}

        result = {
            "fitness": fitness,
            "is_valid": 1,
            "cv_score_std": cv_score_std,
            **secondary,
            **self._bd(ds, model_factory),
        }
        # The fold dispersion estimates uncertainty of the mean score, not of
        # the optional LCB-derived fitness statistic. LCB mode therefore stays
        # honestly unknown until it reports a direct uncertainty estimate.
        return result, _evaluation_measurement_artifact(cv_score_std, len(scores))

    def score_on_test(self, model_factory) -> dict[str, float]:
        ds = self._dataset()
        instance = tabular_metrics.instantiate(model_factory)
        pred = instance.fit_predict(
            ds.X_train.copy(),
            ds.y_train.copy(),
            ds.X_val.copy(),
            ds.y_val.copy(),
            ds.X_test.copy(),
        )
        if ds.task_type == tabular_data.REGRESSION:
            m = tabular_metrics.regression_fold_metrics(ds.y_test, pred)
            return {"test_rmse": m["rmse"], "test_r2": float(r2_score(ds.y_test, pred))}
        if ds.n_classes is None:
            raise ValueError("classification dataset must declare n_classes")
        n_classes = ds.n_classes
        labels = tabular_metrics.to_labels(pred, n_classes)
        acc = float(np.mean(labels == ds.y_test.astype(int)))
        if ds.task_type == tabular_data.BINCLASS:
            proba = tabular_metrics.to_proba(pred, n_classes)
            try:
                auc = float(roc_auc_score(ds.y_test, proba[:, 1]))
            except ValueError:
                auc = 0.5
            return {"test_accuracy": acc, "test_auc": auc}
        macro = float(
            f1_score(ds.y_test.astype(int), labels, average="macro", zero_division=0)
        )
        proba = tabular_metrics.to_proba(pred, n_classes)
        fold = tabular_metrics.classification_fold_metrics(
            ds.y_test, proba, n_classes, ds.task_type
        )
        return {
            "test_accuracy": acc,
            "test_macro_f1": macro,
            "test_log_loss": fold["log_loss"],
        }


def build(name: str) -> TabularProblem:
    return TabularProblem(name)
