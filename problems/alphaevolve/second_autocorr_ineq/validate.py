import numpy as np


def validate(f_values):
    f_values = np.asarray(f_values, dtype=float)

    if f_values.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {f_values.shape}")

    if f_values.size == 0:
        raise ValueError("Array cannot be empty")

    if not np.all(np.isfinite(f_values)):
        raise ValueError("Some values are NaN or infinite")

    if np.any(f_values < -1e-6):
        raise ValueError(
            f"Function must be non-negative. Minimum value: {np.min(f_values):.2e}"
        )

    if np.all(np.abs(f_values) < 1e-12):
        raise ValueError("Function is identically zero (trivial solution)")

    f_nonneg = np.maximum(f_values, 0.0)
    convolution = np.convolve(f_nonneg, f_nonneg, mode="full")

    num_conv_points = len(convolution)
    x_points = np.linspace(-0.5, 0.5, num_conv_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], convolution, [0]))

    l2_norm_squared = 0.0
    for i in range(len(convolution) + 1):
        y1, y2, h = y_points[i], y_points[i + 1], x_intervals[i]
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    norm_1 = np.sum(np.abs(convolution)) / (len(convolution) + 1)
    norm_inf = np.max(np.abs(convolution))
    c2 = l2_norm_squared / (norm_1 * norm_inf)

    if not np.isfinite(c2) or c2 <= 0:
        raise ValueError(f"Invalid C₂ value: {c2}")

    return {
        "fitness": c2,
        "is_valid": 1,
    }
