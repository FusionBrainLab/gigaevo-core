import lightgbm as lgb
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
        model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=0,
            verbosity=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return model.predict(X_query)


def entrypoint() -> type:
    return Model
