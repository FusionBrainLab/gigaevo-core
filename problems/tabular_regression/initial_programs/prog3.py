import numpy as np
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
        model = xgb.XGBRegressor(
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
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model.predict(X_query)


def entrypoint() -> type:
    return Model
