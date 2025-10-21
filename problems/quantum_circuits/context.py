import numpy as np
from typing import List
from helper import Data, ParityMatrix


 
def build_context() -> Data:
    l: List[Data] = []
    name = "binary_addition.npz"
    name = "data/benchmarks_no_gadgets.npz"
    with np.load(name) as f:
        for file in f.files:
            rank = f[file].shape[1]
            n = f[file].shape[2]
            shape = f[file].shape
            # if rank * n < 1000:
            print(file)
            if file == "barenco_toff_5":
                print(f[file])
                l.append(Data(file, 
                              ParityMatrix(f[file][0,:,:].reshape(rank, n).T), 
                              ParityMatrix(np.load("data/barenco_tof_5.matrix.npy")),
                            rank))
    print(np.argwhere(l[0].sota_decomposition.T != l[0].early_decomposition.T).shape[0])
    print(l[0].early_decomposition.P.shape)
    print(l[0].sota_decomposition.P.shape)
    return l[0]

build_context()