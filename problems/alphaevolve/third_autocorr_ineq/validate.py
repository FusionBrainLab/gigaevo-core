import numpy as np


def validate(f_values):
    f_values = np.asarray(f_values, dtype=float)

    if f_values.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {f_values.shape}")

    if f_values.size == 0:
        raise ValueError("Array cannot be empty")

    if not np.all(np.isfinite(f_values)):
        raise ValueError("Some values are NaN or infinite")

    dx = 0.5 / len(f_values)

    integral_f_sq = (np.sum(f_values) * dx) ** 2

    if integral_f_sq < 1e-9:
        raise ValueError("Function integral is close to zero, ratio is unstable.")

    conv = np.convolve(f_values, f_values, mode="full")
    scaled_conv = conv * dx
    max_abs_conv = np.max(np.abs(scaled_conv))

    c3 = max_abs_conv / integral_f_sq

    if not np.isfinite(c3) or c3 <= 0:
        raise ValueError(f"Invalid C₃ value: {c3}")

    return {
        "fitness": c3,
        "is_valid": 1,
    }
