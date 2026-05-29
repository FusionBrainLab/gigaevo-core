import numpy as np
from sklearn.ensemble import RandomForestRegressor


class Model:
    def fit_predict(self, X_train, y_train, X_val, y_val, X_query):
        np.random.seed(0)
        model = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=4)
        model.fit(X_train, y_train)
        return model.predict(X_query)


def entrypoint() -> type:
    return Model
