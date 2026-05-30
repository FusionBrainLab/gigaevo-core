import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=0))]
        )
        pipe.fit(X_train, y_train)
        return pipe.predict(X_query)


def entrypoint() -> type:
    return Model
