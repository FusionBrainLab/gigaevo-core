"""Fair AutoML benchmark on the canonical rtdl California Housing split.

Protocol matches yandex-research/rtdl-revisiting-models:
  - Train data: X_train, y_train
  - Validation data: X_val, y_val  (passed to model for early stopping / HPO,
    same way CatBoost/XGBoost get eval_set in rtdl)
  - Test data: predicted ONLY at end, never seen during fit/HPO

Each library is invoked in a subprocess so a failure / crash in one doesn't
sink the others. Each library gets the same wall-clock budget.

Usage:
    python benchmark_automl.py --lib autogluon --time 1800
    python benchmark_automl.py --lib flaml     --time 1800
    python benchmark_automl.py --lib mljar     --time 1800

Each invocation prints a single line:
    RESULT lib=<name> val_rmse=<f> test_rmse=<f> wall_s=<f>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).parent


def load_split() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    D = HERE / "rtdl_split"
    X_train = np.load(D / "N_train.npy").astype(np.float64)
    y_train = np.load(D / "y_train.npy").astype(np.float64)
    X_val = np.load(D / "N_val.npy").astype(np.float64)
    y_val = np.load(D / "y_val.npy").astype(np.float64)
    X_test = np.load(D / "N_test.npy").astype(np.float64)
    y_test = np.load(D / "y_test.npy").astype(np.float64)
    return X_train, y_train, X_val, y_val, X_test, y_test


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def run_autogluon(time_budget: int) -> tuple[float, float]:
    from autogluon.tabular import TabularPredictor
    import pandas as pd

    X_train, y_train, X_val, y_val, X_test, y_test = load_split()
    cols = [f"x{i}" for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train, columns=cols)
    train_df["y"] = y_train
    val_df = pd.DataFrame(X_val, columns=cols)
    val_df["y"] = y_val
    test_df = pd.DataFrame(X_test, columns=cols)

    save_dir = HERE / "autogluon_bench_artifacts"
    predictor = TabularPredictor(
        label="y",
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=str(save_dir),
        verbosity=2,
    ).fit(
        train_data=train_df,
        tuning_data=val_df,  # explicit holdout — rtdl-style
        use_bag_holdout=True,  # use val for ensembling / model selection
        presets="best_quality",
        time_limit=time_budget,
    )
    val_pred = predictor.predict(val_df.drop(columns="y")).to_numpy()
    test_pred = predictor.predict(test_df).to_numpy()
    return rmse(val_pred, y_val), rmse(test_pred, y_test)


def run_flaml(time_budget: int) -> tuple[float, float]:
    from flaml import AutoML

    X_train, y_train, X_val, y_val, X_test, y_test = load_split()
    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="regression",
        metric="rmse",
        time_budget=time_budget,
        eval_method="holdout",
        estimator_list=[
            "lgbm",
            "xgboost",
            "xgb_limitdepth",
            "catboost",
            "rf",
            "extra_tree",
        ],
        seed=0,
        verbose=2,
    )
    val_pred = automl.predict(X_val)
    test_pred = automl.predict(X_test)
    return rmse(val_pred, y_val), rmse(test_pred, y_test)


def run_mljar(time_budget: int) -> tuple[float, float]:
    """MLJAR-supervised in Compete mode.

    MLJAR cannot accept a pre-built val split without an off-by-one rounding
    bug in its internal split sizer (`train_test_split` raises when
    train_ratio*N + (1-train_ratio)*N round to N+1). We avoid that by feeding
    only X_train to fit; MLJAR runs its own k-fold CV internally for HPO.
    Our held-out X_val is never seen during fit, so val_rmse remains a clean
    out-of-sample number. test_rmse is reported on the canonical test split.
    """
    from supervised.automl import AutoML

    X_train, y_train, X_val, y_val, X_test, y_test = load_split()

    save_dir = HERE / "mljar_bench_artifacts"
    save_dir.mkdir(exist_ok=True)
    automl = AutoML(
        results_path=str(save_dir),
        mode="Compete",
        ml_task="regression",
        eval_metric="rmse",
        total_time_limit=time_budget,
        random_state=0,
    )
    automl.fit(X_train, y_train)
    val_pred = automl.predict(X_val)
    test_pred = automl.predict(X_test)
    return rmse(np.asarray(val_pred), y_val), rmse(np.asarray(test_pred), y_test)


def run_h2o(time_budget: int) -> tuple[float, float]:
    import h2o
    from h2o.automl import H2OAutoML

    X_train, y_train, X_val, y_val, X_test, y_test = load_split()
    cols = [f"x{i}" for i in range(X_train.shape[1])]

    h2o.init(nthreads=-1, max_mem_size="16G")

    def _frame(X: np.ndarray, y: np.ndarray | None = None):
        import pandas as pd

        df = pd.DataFrame(X, columns=cols)
        if y is not None:
            df["y"] = y
        return h2o.H2OFrame(df)

    train_h = _frame(X_train, y_train)
    val_h = _frame(X_val, y_val)
    test_h = _frame(X_test)
    val_no_y = _frame(X_val)

    aml = H2OAutoML(
        max_runtime_secs=time_budget,
        seed=0,
        sort_metric="RMSE",
        stopping_metric="RMSE",
    )
    aml.train(x=cols, y="y", training_frame=train_h, validation_frame=val_h)
    val_pred = aml.leader.predict(val_no_y).as_data_frame().to_numpy().reshape(-1)
    test_pred = aml.leader.predict(test_h).as_data_frame().to_numpy().reshape(-1)
    h2o.cluster().shutdown(prompt=False)
    return rmse(val_pred, y_val), rmse(test_pred, y_test)


LIBS = {
    "autogluon": run_autogluon,
    "flaml": run_flaml,
    "mljar": run_mljar,
    "h2o": run_h2o,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lib", required=True, choices=list(LIBS))
    p.add_argument("--time", type=int, default=1800, help="wall-clock budget (seconds)")
    args = p.parse_args()

    t0 = time.time()
    val_rmse, test_rmse = LIBS[args.lib](args.time)
    wall = time.time() - t0

    print(
        f"\nRESULT lib={args.lib} val_rmse={val_rmse:.5f} test_rmse={test_rmse:.5f} wall_s={wall:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
