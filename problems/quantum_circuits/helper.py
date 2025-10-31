from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


def reconcstruct_from_tensoralpha(tensoralpha_res):
    dim = 3
    factors = [tensoralpha_res[0, :, :].T for r in range(dim)]
    letters = "".join(chr(97 + i) for i in range(dim))
    spec = ",".join(f"{chr(97 + i)}r" for i in range(dim)) + f"->{letters}r"
    and_per_r = jnp.einsum(spec, *factors, optimize=True).astype(jnp.uint8)
    T = (jnp.sum(and_per_r, axis=-1) & jnp.uint8(1)).astype(jnp.uint8)
    return T


def reconstruct_from_single_binary_factor(f: jnp.ndarray) -> jnp.ndarray:
    f = f.astype(jnp.uint8)
    return jnp.einsum("a,b,c->abc", *(f, f, f)).astype(jnp.uint8)


@dataclass
class Data:
    name: str
    tensor: jnp.ndarray
    sota_rank: int


def reconstruct_from_multi_binary_factors(b: jnp.ndarray) -> jnp.ndarray:
    spec = "ar,br,cr->abcr"
    and_per_r = jnp.einsum(spec, b, b, b).astype(jnp.uint8)
    return (jnp.sum(and_per_r, axis=-1) & jnp.uint8(1)).astype(jnp.uint8)


def get_residual_num(T1: jnp.ndarray, T2: jnp.ndarray = None):
    if T2 is None:
        return int(jnp.sum(T1))
    return jnp.sum(T1 ^ T2)


# это для теста нужно было
def matmul_tensor_4x4():
    n = 4
    dim = n * n
    T = jnp.zeros((dim, dim, dim), dtype=jnp.int8)
    idx = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a = i * n + j
                b = j * n + k
                c = i * n + k
                idx.append((a, b, c))
    a, b, c = zip(*idx)
    T = T.at[(jnp.array(a), jnp.array(b), jnp.array(c))].set(1)
    return T


def generate_bool_tensor(
    shape: Tuple[int, ...],
    rank: int,
    *,
    key: jax.Array = jax.random.PRNGKey(0),
    p: float = 0.5,
    return_factors: bool = True,
):
    N = len(shape)
    keys = jax.random.split(key, N)

    factors = tuple(
        jax.random.bernoulli(keys[n], p=p, shape=(shape[n], rank)).astype(jnp.uint8)
        for n in range(N)
    )

    letters = "".join(chr(97 + i) for i in range(N))
    spec = ",".join(f"{chr(97 + i)}r" for i in range(N)) + f"->{letters}r"
    and_per_r = jnp.einsum(spec, *factors, optimize=True).astype(jnp.uint8)
    T = (jnp.sum(and_per_r, axis=-1) % 2).astype(jnp.uint8)
    return (T, factors) if return_factors else T


def generate_symmetric_bool_tensor(
    N: int,
    dim: int,
    rank: int,
    *,
    key: jax.Array = jax.random.PRNGKey(0),
    p: float = 0.5,
    return_factors: bool = True,
):
    keys = jax.random.split(key, dim)

    factor = jax.random.bernoulli(keys[0], p=p, shape=(N, rank)).astype(jnp.uint8)
    factors = tuple(factor for _ in range(dim))
    letters = "".join(chr(97 + i) for i in range(dim))
    spec = ",".join(f"{chr(97 + i)}r" for i in range(dim)) + f"->{letters}r"
    and_per_r = jnp.einsum(spec, *factors, optimize=True).astype(jnp.uint8)

    T = (jnp.sum(and_per_r, axis=-1) % 2).astype(jnp.uint8)

    return (T, factors) if return_factors else T
