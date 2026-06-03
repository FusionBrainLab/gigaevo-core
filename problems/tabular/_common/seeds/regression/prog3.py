import numpy as np
from xgboost import XGBRegressor


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        params = dict(
            learning_rate=0.05,
            max_depth=6,
            tree_method="hist",
            random_state=0,
            n_jobs=4,
            verbosity=0,
        )
        search = XGBRegressor(n_estimators=2000, early_stopping_rounds=50, **params)
        search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best = (search.best_iteration or 0) + 1
        final = XGBRegressor(n_estimators=best, **params)
        final.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
        return final.predict(X_query)


def entrypoint() -> type:
    return Model
