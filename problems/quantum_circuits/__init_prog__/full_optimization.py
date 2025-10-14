from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.nn import initializers, sigmoid
import optax
from dataclasses import dataclass
from typing import Tuple, List, Any
from helper import generate_symmetric_bool_tensor


def entrypoint(context: List[jnp.ndarray]):
    def reconstruct_tensor_from_factors(factors: jnp.ndarray, dim: int) -> jnp.ndarray:
        letters = ''.join(chr(97 + i) for i in range(dim))
        spec = ','.join(f"{chr(97+i)}r" for i in range(dim)) + "->" + letters + "r"
        probs = tuple(sigmoid(factors) for _ in range(dim))
        and_per_r = jnp.einsum(spec, *probs)          # (..., R)
        return 0.5 - 0.5 * jnp.prod(1.0 - 2 * and_per_r, axis=-1)

    @jax.jit
    def bce_loss(target: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
        eps = 1e-6
        return -jnp.sum(target * jnp.log(P + eps)  + (1.0 - target) * jnp.log(1 - P + eps))

    @dataclass
    class CPHyperParams:
        learning_rate: float = 3e-2
        init_scale: float = 1e+0

    class WarDecFinder:
        def __init__(
            self,
            target: jnp.ndarray,
            rank: int=None,
            key: jax.random.PRNGKey = jax.random.PRNGKey(0),
            dtype: Any = jnp.float32,
            hypers: CPHyperParams | None = None,
            opt_state: optax.OptState | None = None,
        ):
            self.target = target.astype(jnp.float32)
            self.N = self.target.shape[0]
            if rank is None: 
                #TODO
                pass
            else:
                self.rank = rank
            self.dim = len(target.shape)
            self.dtype = dtype
            self.hypers = hypers or CPHyperParams()
            self.factors = self.generate_init_factors(key)

            self.opt_state = opt_state or self.optimizer.init(self.factors)
            self._build_train_step()

        def define_optimizer(self):
            self.optimizer = optax.adam(self.hypers.learning_rate)

        def generate_init_factors(self, key):
            init = initializers.normal(self.hypers.init_scale, self.dtype)
            return init(key, (self.N, self.rank), dtype=self.dtype)

        def _build_train_step(self) -> None:
            optimizer = self.optimizer
            target = self.target
            dim = self.dim
            def loss_fn(factors: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
                P = reconstruct_tensor_from_factors(factors, dim)
                return bce_loss(target, P)

            @jax.jit
            def train_step(factors: Tuple[jnp.ndarray, ...], opt_state: optax.OptState):
                grads = jax.grad(loss_fn)(factors)
                updates, opt_state = optimizer.update(grads, opt_state, factors)
                factors = optax.apply_updates(factors, updates)
                return factors, opt_state

            self.train_step = train_step

        def loss(self) -> jnp.ndarray:
            P = reconstruct_tensor_from_factors(self.factors, self.dim)
            return bce_loss(self.target, P, self.hypers.alpha_pos, self.hypers.beta_neg)

        def fit(self, steps: int=50000) -> None:
            for s in range(1, steps + 1):
                self.step()

        def step(self) -> None:
            self.factors, self.opt_state = self.train_step(self.factors, self.opt_state)

        def reconstruct_bool(self) -> jnp.ndarray:
            B = reconstruct_tensor_from_factors(self.factors, self.dim)
            return (B > 0.5).astype(jnp.int32)
    solutions = []
    for tensor in context:
        wdf = WarDecFinder(tensor)
        wdf.fit()
        solutions.append(jnp.sum(tensor - reconstruct_tensor_from_factors(wdf.factors, dim=len(tensor.shape)) > 0.1))

    return solutions
