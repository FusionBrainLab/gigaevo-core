import jax
import jax.numpy as jnp


def compute_c2(f_values: jnp.ndarray) -> float:
    f_non_negative = jax.nn.relu(f_values)

    N = len(f_values)
    padded_f = jnp.pad(f_non_negative, (0, N))
    fft_f = jnp.fft.fft(padded_f)
    convolution = jnp.fft.ifft(fft_f * fft_f).real

    num_conv_points = len(convolution)
    h = 1.0 / (num_conv_points + 1)

    y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
    y1, y2 = y_points[:-1], y_points[1:]

    l2_norm_squared = jnp.sum((h / 3) * (y1**2 + y1 * y2 + y2**2))

    norm_1 = jnp.sum(jnp.abs(convolution)) / (len(convolution) + 1)

    norm_inf = jnp.max(jnp.abs(convolution))

    denominator = norm_1 * norm_inf
    c2_ratio = l2_norm_squared / denominator

    return c2_ratio
