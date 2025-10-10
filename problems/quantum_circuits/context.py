import numpy as np
from helper import reconcstruct_from_tensoralpha, Data


def build_context() -> list[Data]:
    l = []
    name = "binary_addition.npz"
    name = "data/benchmarks_gadgets.npz"
    with np.load(name) as f:
        for file in f.files:
            rank = f[file].shape[1]
            n = f[file].shape[2]
            if rank * n < 200:
                l.append(Data(file, reconcstruct_from_tensoralpha(f[file]), rank))

    return l[:]