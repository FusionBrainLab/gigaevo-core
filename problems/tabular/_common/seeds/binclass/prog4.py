import lightgbm as lgb
import numpy as np


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        n_classes = int(max(y_train.max(), y_val.max())) + 1
        params = dict(
            learning_rate=0.05,
            num_leaves=63,
            random_state=0,
            n_jobs=4,
            verbose=-1,
        )
        search = lgb.LGBMClassifier(n_estimators=2000, **params)
        search.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best = search.best_iteration_ or 2000
        clf = lgb.LGBMClassifier(n_estimators=best, **params)
        clf.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
        full = np.zeros((X_query.shape[0], n_classes))
        full[:, clf.classes_.astype(int)] = clf.predict_proba(X_query)
        return full


def entrypoint() -> type:
    return Model
