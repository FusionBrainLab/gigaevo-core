from catboost import CatBoostRegressor
import numpy as np
from sklearn.cluster import KMeans


class Model:
    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        np.random.seed(0)

        lat_lon_train = X_train[:, [6, 7]]
        lat_lon_val = X_val[:, [6, 7]]
        lat_lon_query = X_query[:, [6, 7]]

        km = KMeans(n_clusters=20, random_state=0, n_init=10)
        km.fit(lat_lon_train)

        def augment(X: np.ndarray, lat_lon: np.ndarray) -> np.ndarray:
            distances = km.transform(lat_lon)
            cluster_id = km.predict(lat_lon).reshape(-1, 1).astype(float)
            return np.hstack([X, distances, cluster_id])

        X_train_aug = augment(X_train, lat_lon_train)
        X_val_aug = augment(X_val, lat_lon_val)
        X_query_aug = augment(X_query, lat_lon_query)

        model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            random_seed=0,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
            early_stopping_rounds=50,
        )
        model.fit(X_train_aug, y_train, eval_set=(X_val_aug, y_val))
        return np.asarray(model.predict(X_query_aug), dtype=float)


def entrypoint() -> type:
    return Model
