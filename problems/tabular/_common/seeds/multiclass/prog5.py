from catboost import CatBoostClassifier
import numpy as np


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        n_classes = int(max(y_train.max(), y_val.max())) + 1
        clf = CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=6,
            random_seed=0,
            thread_count=4,
            logging_level="Silent",
        )
        clf.fit(X_train, y_train)
        full = np.zeros((X_query.shape[0], n_classes))
        full[:, clf.classes_.astype(int)] = clf.predict_proba(X_query)
        return full


def entrypoint() -> type:
    return Model
