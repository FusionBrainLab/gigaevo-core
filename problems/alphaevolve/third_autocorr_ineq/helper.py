import jax
import jax.numpy as jnp


def compute_c3(f_values: jnp.ndarray) -> float:
    domain_width = 0.5
    dx = domain_width / len(f_values)

    integral_f = jnp.sum(f_values) * dx
    eps = 1e-9
    integral_f_sq_safe = jnp.maximum(integral_f**2, eps)

    N = len(f_values)
    padded_f = jnp.pad(f_values, (0, N))
    fft_f = jnp.fft.fft(padded_f)
    conv_f_f = jnp.fft.ifft(fft_f * fft_f).real

    scaled_conv_f_f = conv_f_f * dx
    max_abs_conv = jnp.max(jnp.abs(scaled_conv_f_f))

    c3_ratio = max_abs_conv / integral_f_sq_safe

    return c3_ratio
