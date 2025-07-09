import numpy as np

def validate(payload: tuple[dict[str, np.ndarray], dict[str, np.ndarray]]) -> dict[str, float]:
    # payload = tuple[context, output of program]
    context, y_pred = payload
    y_true = context["y_test"]
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} != {y_pred.shape}")
    return {"fitness": -np.mean((y_pred - y_true)**2), "is_valid": 1}