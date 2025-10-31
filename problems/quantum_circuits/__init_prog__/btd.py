from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from helper import generate_bool_tensor
import jax
from jax.nn import initializers, sigmoid
import jax.numpy as jnp
import optax


def reconstruct_prob_from_factors(factors: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    N = len(factors)
    letters = "".join(chr(97 + i) for i in range(N))
    spec = ",".join(f"{chr(97 + i)}r" for i in range(N)) + "->" + letters + "r"
    probs = tuple(sigmoid(F) for F in factors)
    and_per_r = jnp.einsum(spec, *probs)  # (..., R)
    return 0.5 - 0.5 * jnp.prod(1.0 - 2 * and_per_r, axis=-1)


def bce_loss(
    target: jnp.ndarray, P: jnp.ndarray, alpha: float, beta: float
) -> jnp.ndarray:
    eps = 1e-6
    return -jnp.sum(
        alpha * target * jnp.log(P + eps) + beta * (1.0 - target) * jnp.log(1 - P + eps)
    )


@dataclass
class CPHyperParams:
    learning_rate: float = 1e-1
    init_scale: float = 1e1
    alpha_pos: float = 1.0
    beta_neg: float = 1.0
    threshold: float = 0.5


class TDFinder:
    def __init__(
        self,
        target: jnp.ndarray,
        rank: int,
        key: jax.random.PRNGKey = jax.random.PRNGKey(0),
        dtype: Any = jnp.float32,
        hypers: CPHyperParams | None = None,
        factors: Tuple[jnp.ndarray, ...] | None = None,
        optimizer: optax.GradientTransformation | None = None,
        opt_state: optax.OptState | None = None,
    ):
        self.target = target.astype(jnp.float32)
        self.rank = rank
        self.key = key
        self.dtype = dtype
        self.hypers = hypers or CPHyperParams()
        self.init = initializers.normal(self.hypers.init_scale, self.dtype)

        shape = self.target.shape
        N = len(shape)
        if factors is None:
            keys = jax.random.split(self.key, N)
            self.factors = tuple(
                self.init(keys[i], (shape[i], self.rank), dtype=self.dtype)
                for i in range(N)
            )
        else:
            self.factors = factors

        self.optimizer = optimizer or optax.adam(self.hypers.learning_rate)
        self.opt_state = opt_state or self.optimizer.init(self.factors)
        self._build_train_step()

    def _build_train_step(self) -> None:
        optimizer = self.optimizer
        target = self.target
        alpha = self.hypers.alpha_pos
        beta = self.hypers.beta_neg

        def loss_fn(factors: Tuple[jnp.ndarray, ...]) -> jnp.ndarray:
            P = reconstruct_prob_from_factors(factors)
            return bce_loss(target, P, alpha, beta)

        @jax.jit
        def train_step(factors: Tuple[jnp.ndarray, ...], opt_state: optax.OptState):
            grads = jax.grad(loss_fn)(factors)
            updates, opt_state = optimizer.update(grads, opt_state, factors)
            factors = optax.apply_updates(factors, updates)
            return factors, opt_state

        self.train_step = train_step

    def reconstruct_prob(self) -> jnp.ndarray:
        return reconstruct_prob_from_factors(self.factors)

    def loss(self) -> jnp.ndarray:
        P = self.reconstruct_prob()
        return bce_loss(self.target, P, self.hypers.alpha_pos, self.hypers.beta_neg)

    def step(self) -> None:
        self.factors, self.opt_state = self.train_step(self.factors, self.opt_state)

    def fit(self, steps: int, log_every: int = 0) -> None:
        for s in range(1, steps + 1):
            self.step()
            if log_every and (s % log_every == 0 or s == 1 or s == steps):
                current_loss = self.loss()
                print(f"[step {s}] loss={float(current_loss):.6f}")

    def reconstruct_bool(self) -> jnp.ndarray:
        B = reconstruct_prob_from_factors(self.factors)
        return (B > 0.5).astype(jnp.int32)


if __name__ == "__main__":
    r = 5
    N = 8
    shape = (N, N, N)
    A, _ = generate_bool_tensor(shape, r)
    tdf = TDFinder(A, r)
    tdf.fit(50000, log_every=5000)
    A_hat_prob = tdf.reconstruct_prob()
    A_hat_bin = reconstruct_prob_from_factors(tdf.factors)
    A_hat_bin = tdf.reconstruct_bool()
    print("prob mean:", float(A_hat_prob.mean()), "bin mean:", float(A_hat_bin.mean()))
    print(f"num of errors: {jnp.sum(A - A_hat_bin > 0.1)}")
