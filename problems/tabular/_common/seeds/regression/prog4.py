from lightgbm import LGBMRegressor
import numpy as np


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            random_state=0,
            n_jobs=4,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        return model.predict(X_query)


def entrypoint() -> type:
    return Model
