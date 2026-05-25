import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0, random_state=0)),
            ]
        )
        pipe.fit(X_train, y_train)
        return pipe.predict(X_query)


def entrypoint() -> type:
    return Model
