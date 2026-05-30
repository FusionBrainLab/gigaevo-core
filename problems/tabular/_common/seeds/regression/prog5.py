from catboost import CatBoostRegressor
import numpy as np


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        model = CatBoostRegressor(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            random_seed=0,
            thread_count=4,
            logging_level="Silent",
        )
        model.fit(X_train, y_train)
        return model.predict(X_query)


def entrypoint() -> type:
    return Model
