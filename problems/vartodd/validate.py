from typing import List, Tuple

import numpy as np
from helper import Matrix, Tensor3D, get_matrix


def validate(
    result: Tuple[np.ndarray, str]
) -> dict[str, float]:
    context = get_matrix()
    result, report, best_path = result
    if np.any(Tensor3D(context) != Tensor3D(Matrix.from_numpy(result))):
        return {"fitness": float('inf'),
                "is_valid": 0.0,
                "aux info": report + best_path,
                }
    
    return {"fitness": result.shape[0] + np.sum(result) / np.size(result), 
            "is_valid": 1.0,
            "aux info": report + best_path,
            }
