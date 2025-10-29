import numpy as np


def validate(
    context, y_pred,
) -> dict[str, float]:
    y_true = context["y_test"]
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} != {y_pred.shape}")
    return {"fitness": -np.mean((y_pred - y_true) ** 2), "is_valid": 1}
