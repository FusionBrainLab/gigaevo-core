import numpy as np
from sklearn.linear_model import LinearRegression

def entrypoint(context: dict[str, np.ndarray]) -> np.ndarray:
    """
    Fit a linear regression model and predict the target values.
    """
    X_train = context["X_train"]
    y_train = context["y_train"]
    X_test = context["X_test"]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred