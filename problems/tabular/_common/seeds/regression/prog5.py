from catboost import CatBoostRegressor
import numpy as np


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        params = dict(
            learning_rate=0.05,
            depth=6,
            random_seed=0,
            thread_count=4,
            logging_level="Silent",
        )
        search = CatBoostRegressor(iterations=2000, early_stopping_rounds=50, **params)
        search.fit(X_train, y_train, eval_set=(X_val, y_val))
        best = search.get_best_iteration() + 1
        final = CatBoostRegressor(iterations=best, **params)
        final.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
        return final.predict(X_query)


def entrypoint() -> type:
    return Model
