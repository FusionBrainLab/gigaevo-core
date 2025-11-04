import numpy as np
from helper import Data, Matrix, Tensor3D

def validate(
    payload: tuple[Data, np.ndarray],
) -> dict[str, float]:
    context, result = payload
    if np.any(Tensor3D(Matrix.from_numpy(context.sota_decomposition)) != Tensor3D(Matrix.from_numpy(result))):
        return {"fitness": -result.shape[0] + context.sota_rank, "is_valid": 0}
    return {"fitness": -result.shape[0] + context.sota_rank, "is_valid": 1}
