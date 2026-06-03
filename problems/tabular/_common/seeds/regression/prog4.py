import lightgbm as lgb
import numpy as np


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        params = dict(
            learning_rate=0.05,
            num_leaves=63,
            random_state=0,
            n_jobs=4,
            verbose=-1,
        )
        search = lgb.LGBMRegressor(n_estimators=2000, **params)
        search.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best = search.best_iteration_ or 2000
        final = lgb.LGBMRegressor(n_estimators=best, **params)
        final.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
        return final.predict(X_query)


def entrypoint() -> type:
    return Model
