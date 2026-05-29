import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        n_classes = int(max(y_train.max(), y_val.max())) + 1
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=0)),
            ]
        )
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_query)
        full = np.zeros((X_query.shape[0], n_classes))
        full[:, pipe.classes_.astype(int)] = proba
        return full


def entrypoint() -> type:
    return Model
