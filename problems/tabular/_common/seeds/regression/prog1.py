import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        X = np.concatenate([X_train, X_val])
        y = np.concatenate([y_train, y_val])
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("spline", SplineTransformer(n_knots=5, degree=3)),
                ("reg", RidgeCV(alphas=np.logspace(-3, 3, 13))),
            ]
        )
        pipe.fit(X, y)
        return pipe.predict(X_query)


def entrypoint() -> type:
    return Model
