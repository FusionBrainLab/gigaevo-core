import numpy as np


def compute_c(h_values):
    h_values = np.asarray(h_values, dtype=float)
    N = len(h_values)
    dx = 2.0 / N
    
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c = np.max(correlation)
    
    return c


def validate(h_values):
    h_values = np.asarray(h_values, dtype=float)
    
    if h_values.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {h_values.shape}")
    
    if h_values.size == 0:
        raise ValueError("Array cannot be empty")
    
    if not np.all(np.isfinite(h_values)):
        raise ValueError("Some values are NaN or infinite")
    
    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(
            f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]"
        )

    domain_width = 2.0
    dx = domain_width / len(h_values)
    integral = np.sum(h_values) * dx
    
    if not np.isclose(integral, 1.0, atol=1e-3):
        raise ValueError(
            f"Normalization constraint violated. ∫₀² h(x)dx = {integral:.6f}, expected 1.0"
        )

    try:
        c = compute_c(h_values)
    except Exception as e:
        raise ValueError(f"Error computing C: {e}")
    
    if not np.isfinite(c) or c <= 0:
        raise ValueError(f"Invalid C value: {c}")
    
    return {"fitness": c, "is_valid": 1}
