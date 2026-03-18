from typing import List, Tuple
import re

import numpy as np
from helper import Matrix, Tensor3D, get_matrix


def validate(
    result: Tuple[np.ndarray, str, str]
) -> dict[str, float]:
    context = get_matrix()
    result, report, best_path = result
    if np.any(Tensor3D(context) != Tensor3D(Matrix.from_numpy(result))):
        return {"fitness": float('inf'),
                "is_valid": 0.0,
                "aux info": report + best_path,
                }
    
    base_fitness = result.shape[0] + np.sum(result) / np.size(result)
    penalty = 0.0
    m = re.search(r"\bloaded_rank:\s*(\d+)", report)
    if m is not None:
        loaded_rank = int(m.group(1))
        found_rank = int(result.shape[0])
        if found_rank >= loaded_rank:
            penalty = 10.0 + float(found_rank - loaded_rank)
    return {"fitness": base_fitness + penalty, 
            "is_valid": 1.0,
            "aux info": report + best_path,
            }
