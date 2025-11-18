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

    domain_width = 0.5
    dx = domain_width / len(f_values)

    f_nonneg = np.maximum(f_values, 0.0)

    integral_f = np.sum(f_nonneg) * dx

    if integral_f**2 < 1e-9:
        raise ValueError("Function integral is close to zero, ratio is unstable.")

    N = len(f_values)
    padded_f = np.pad(f_nonneg, (0, N))
    fft_f = np.fft.fft(padded_f)
    conv_f_f = np.fft.ifft(fft_f * fft_f).real

    scaled_conv = conv_f_f * dx
    max_conv = np.max(scaled_conv)

    c1 = max_conv / (integral_f**2)

    if not np.isfinite(c1) or c1 <= 0:
        raise ValueError(f"Invalid C₁ value: {c1}")

    return {"fitness": c1, "is_valid": 1}
