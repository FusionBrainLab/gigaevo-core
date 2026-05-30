import numpy as np
from xgboost import XGBClassifier


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        n_classes = int(max(y_train.max(), y_val.max())) + 1
        clf = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            tree_method="hist",
            random_state=0,
            n_jobs=4,
            verbosity=0,
        )
        clf.fit(X_train, y_train)
        full = np.zeros((X_query.shape[0], n_classes))
        full[:, clf.classes_.astype(int)] = clf.predict_proba(X_query)
        return full


def entrypoint() -> type:
    return Model
