from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from gf2lib.gf2lib import Matrix, ToddGenerator, TohpeGenerator, Tensor3D, eliminate_duplications_inplace, gauss_elimination_inplace, NullSpace

@dataclass
class Data:
    name: str
    sota_decomposition: np.ndarray
    early_decomposition: np.ndarray
    sota_rank: int