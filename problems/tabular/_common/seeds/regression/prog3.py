import numpy as np
from xgboost import XGBRegressor


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            tree_method="hist",
            random_state=0,
            n_jobs=4,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        return model.predict(X_query)


def entrypoint() -> type:
    return Model
