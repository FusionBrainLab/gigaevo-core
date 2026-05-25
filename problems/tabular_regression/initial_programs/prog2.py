import numpy as np
from sklearn.ensemble import RandomForestRegressor


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
        model = RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=0,
        )
        model.fit(X_train, y_train)
        return model.predict(X_query)


def entrypoint() -> type:
    return Model
