import jax
import jax.numpy as jnp


def compute_c1(f_values: jnp.ndarray) -> float:
    domain_width = 0.5
    dx = domain_width / len(f_values)

    f_non_negative = jax.nn.relu(f_values)

    integral_f = jnp.sum(f_non_negative) * dx
    eps = 1e-9
    integral_f_sq_safe = jnp.maximum(integral_f**2, eps)

    N = len(f_values)
    padded_f = jnp.pad(f_non_negative, (0, N))
    fft_f = jnp.fft.fft(padded_f)
    conv_f_f = jnp.fft.ifft(fft_f * fft_f).real

    scaled_conv = conv_f_f * dx
    max_conv = jnp.max(scaled_conv)

    c1_ratio = max_conv / integral_f_sq_safe

    return c1_ratio
