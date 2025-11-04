import numpy as np
from typing import List
from helper import Data, Tensor3D, Matrix, eliminate_duplications_inplace 


def build_context() -> Data:
    l: List[Data] = []
    name = "binary_addition.npz"
    name = "data/benchmarks_no_gadgets.npz"
    name = "data/multiplication_finite_fields_no_gadgets.npz"
    with np.load(name) as f:
        for file in f.files:
            rank = f[file].shape[1]
            n = f[file].shape[2]
            shape = f[file].shape
            # if rank * n < 1000:
            print(file)
            name = "gf_2pow6_mult_comp1"
            if file == name:
                print(f[file])
                l.append(Data(file, 
                              Matrix.from_numpy(f[file][0,:,:].reshape(rank, n)), 
                              Matrix.from_numpy(np.load(f"data/{name}.matrix.npy").T),
                            rank))
                # eliminate_duplications_inplace(l[-1].early_decomposition)
    # print(np.argwhere(l[0].sota_decomposition.T != l[0].early_decomposition.T).shape)
    return l[0]

# build_context()
