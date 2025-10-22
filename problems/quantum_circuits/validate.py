import numpy as np
import jax.numpy as jnp
from helper import Data, ParityMatrix


def validate(
    payload: tuple[Data, ParityMatrix],
) -> dict[str, float]:
    context, result = payload
    if np.any(context.sota_decomposition.T != result.to_symmetric_tensor()):
        return {"fitness": -result.P.shape[1] + context.sota_rank, "is_valid": 0}
    return {"fitness": -result.P.shape[1] + np.random.uniform(0, 0.8) + context.sota_rank, "is_valid": 1}