import jax
import jax.numpy as jnp


def compute_c(h_values: jnp.ndarray) -> jnp.ndarray:
    h_values = jnp.asarray(h_values, dtype=jnp.float32)
    N = len(h_values)
    dx = 2.0 / N
    
    j_values = 1.0 - h_values
    padded_h = jnp.pad(h_values, (0, N))
    padded_j = jnp.pad(j_values, (0, N))
    
    fft_h = jnp.fft.fft(padded_h)
    fft_j = jnp.fft.fft(padded_j)
    correlation = jnp.fft.ifft(fft_h * jnp.conj(fft_j)).real * dx
    
    c = jnp.max(correlation)
    
    return c
