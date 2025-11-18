import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def build_context() -> dict[str, np.ndarray]:
    housing = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing[0], housing[1], test_size=0.2, random_state=42
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
