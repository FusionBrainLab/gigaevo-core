import numpy as np
from helper import Data, Matrix, Tensor3D
from context import build_context


def validate(
    context: Data,
    result: np.ndarray
) -> dict[str, float]:
    if np.any(Tensor3D(Matrix.from_numpy(context.sota_decomposition)) != Tensor3D(Matrix.from_numpy(result))):
        return {"fitness": -result.shape[0] + context.sota_rank, "is_valid": 0}
    return {"fitness": -result.shape[0] + context.sota_rank, "is_valid": 1}
