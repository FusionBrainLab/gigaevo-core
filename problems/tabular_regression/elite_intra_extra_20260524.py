"""
Rescued elite from tabular_regression intra_extra_memory run (db=15, output dir
tabular_regression_intra_extra_20260523_161718).

Program id: f468603d-c19, gen 17.
Val RMSE under old single-split protocol: 0.40737
Test RMSE (score_test.py):                 0.38013

Run elapsed at extraction: +20h45m (budget reached at ~800 unique evals).
Redis db=15 was accidentally flushed after the run was stopped, so this file
is the only surviving artifact for this program.

The original program had its own 5-fold CV stacking (KFold over X_train_aug,
OOF CatBoost+KNN predictions, RidgeCV meta-learner). With the framework now
running 5-fold CV at the validate() layer, that inner CV becomes nested CV —
5x duplicate compute, no robustness gain, and a TLE risk. This file keeps the
feature-engineering spine and the CatBoost+KNN blend but drops the inner CV:
a single CatBoost fit with eval_set early stopping, a single KNN, and a
fixed-weight blend (0.7 CatBoost / 0.3 KNN ≈ the RidgeCV solution observed on
this dataset).
"""

from catboost import CatBoostRegressor
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor


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

        def to_3d(lat_lon):
            lat_rad = np.radians(lat_lon[:, 0])
            lon_rad = np.radians(lat_lon[:, 1])
            x = np.cos(lat_rad) * np.cos(lon_rad)
            y = np.cos(lat_rad) * np.sin(lon_rad)
            z = np.sin(lat_rad)
            return np.column_stack((x, y, z))

        lat_lon_train = X_train[:, [6, 7]]
        lat_lon_val = X_val[:, [6, 7]]
        lat_lon_query = X_query[:, [6, 7]]
        lat_lon_train_3d = to_3d(lat_lon_train)
        lat_lon_query_3d = to_3d(lat_lon_query)

        km = KMeans(n_clusters=50, random_state=0, n_init=10)
        km.fit(lat_lon_train_3d)

        def augment(X: np.ndarray, lat_lon: np.ndarray) -> np.ndarray:
            lat_lon_3d = to_3d(lat_lon)
            distances = km.transform(lat_lon_3d)
            cluster_id = km.predict(lat_lon_3d).reshape(-1, 1).astype(float)
            room_bedroom_ratio = (X[:, 2] / (X[:, 3] + 1e-5)).reshape(-1, 1)
            rooms_per_person = (X[:, 2] / (X[:, 5] + 1e-5)).reshape(-1, 1)
            bedrooms_per_person = (X[:, 3] / (X[:, 5] + 1e-5)).reshape(-1, 1)
            household_count = (X[:, 4] / (X[:, 5] + 1e-5)).reshape(-1, 1)
            return np.hstack(
                [
                    X,
                    distances,
                    cluster_id,
                    room_bedroom_ratio,
                    rooms_per_person,
                    bedrooms_per_person,
                    household_count,
                ]
            )

        X_train_aug = augment(X_train, lat_lon_train)
        X_val_aug = augment(X_val, lat_lon_val)
        X_query_aug = augment(X_query, lat_lon_query)

        cat_model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=1.5,
            loss_function="RMSE",
            random_seed=0,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
            early_stopping_rounds=150,
        )
        cat_model.fit(X_train_aug, y_train, eval_set=(X_val_aug, y_val))

        knn_model = KNeighborsRegressor(n_neighbors=7, weights="distance")
        knn_model.fit(lat_lon_train_3d, y_train)

        cat_pred = cat_model.predict(X_query_aug)
        knn_pred = knn_model.predict(lat_lon_query_3d)
        blended = 0.7 * cat_pred + 0.3 * knn_pred

        return np.clip(blended, 0.15, 5.0)


def entrypoint() -> type:
    return Model
