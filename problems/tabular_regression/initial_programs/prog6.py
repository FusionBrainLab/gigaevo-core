from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize
import xgboost as xgb


class Model:
    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        np.random.seed(0)

        xgb_m = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=0,
            verbosity=0,
            early_stopping_rounds=50,
        )
        xgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        lgb_m = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=0,
            verbosity=-1,
        )
        lgb_m.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        cb_m = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            random_seed=0,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
            early_stopping_rounds=50,
        )
        cb_m.fit(X_train, y_train, eval_set=(X_val, y_val))

        val_preds = np.stack(
            [
                xgb_m.predict(X_val),
                lgb_m.predict(X_val),
                np.asarray(cb_m.predict(X_val), dtype=float),
            ],
            axis=1,
        )

        def loss(w_raw: np.ndarray) -> float:
            w = np.clip(w_raw, 0.0, None)
            s = w.sum()
            if s <= 0:
                return float("inf")
            w = w / s
            return float(np.sqrt(np.mean((val_preds @ w - y_val) ** 2)))

        res = minimize(loss, x0=np.ones(3) / 3.0, method="Nelder-Mead")
        w = np.clip(res.x, 0.0, None)
        w = w / w.sum()

        query_preds = np.stack(
            [
                xgb_m.predict(X_query),
                lgb_m.predict(X_query),
                np.asarray(cb_m.predict(X_query), dtype=float),
            ],
            axis=1,
        )
        return query_preds @ w


def entrypoint() -> type:
    return Model
