import numpy as np
from sklearn.ensemble import RandomForestClassifier


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        n_classes = int(max(y_train.max(), y_val.max())) + 1
        X = np.concatenate([X_train, X_val])
        y = np.concatenate([y_train, y_val])
        clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4)
        clf.fit(X, y)
        full = np.zeros((X_query.shape[0], n_classes))
        full[:, clf.classes_.astype(int)] = clf.predict_proba(X_query)
        return full


def entrypoint() -> type:
    return Model
