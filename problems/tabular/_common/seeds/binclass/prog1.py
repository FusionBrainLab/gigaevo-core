import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        n_classes = int(max(y_train.max(), y_val.max())) + 1
        X = np.concatenate([X_train, X_val])
        y = np.concatenate([y_train, y_val])
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("spline", SplineTransformer(n_knots=5, degree=3)),
                ("clf", LogisticRegression(max_iter=2000, random_state=0)),
            ]
        )
        pipe.fit(X, y)
        full = np.zeros((X_query.shape[0], n_classes))
        full[:, pipe.classes_.astype(int)] = pipe.predict_proba(X_query)
        return full


def entrypoint() -> type:
    return Model
