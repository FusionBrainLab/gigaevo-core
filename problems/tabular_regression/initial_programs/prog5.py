from catboost import CatBoostRegressor
import numpy as np


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
        model = CatBoostRegressor(
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
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        return np.asarray(model.predict(X_query), dtype=float)


def entrypoint() -> type:
    return Model
