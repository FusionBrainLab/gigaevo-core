from lightgbm import LGBMClassifier
import numpy as np


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        n_classes = int(max(y_train.max(), y_val.max())) + 1
        clf = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            random_state=0,
            n_jobs=4,
            verbose=-1,
        )
        clf.fit(X_train, y_train)
        full = np.zeros((X_query.shape[0], n_classes))
        full[:, clf.classes_.astype(int)] = clf.predict_proba(X_query)
        return full


def entrypoint() -> type:
    return Model
