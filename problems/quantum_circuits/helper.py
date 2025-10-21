from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Data:
    name: str
    sota_decomposition: ParityMatrix
    early_decomposition: ParityMatrix
    sota_rank: int
    
class ParityMatrix:
    P: np.ndarray 
    T: np.ndarray
    def __init__(self, P: np.ndarray):
        if len(P.shape) == 2:
            self.P = P.astype(np.bool)
            self.T = self.to_symmetric_tensor()

    def num_factors(self) -> int:
        return self.P.shape[1]
    
    def get_all_qubits(self) -> int:
        return self.P.shape[0]

    def get_target_qubits(self) -> int:
        return self.P.shape[0]
        
    def get_factor(self, i: int) -> int:
        """return column"""
        return self.P[:, i]

    def get_row(self, i: int):
        return self.P[i,:]
    
    def get_target_tensor(self):
        return self.T
    
    def to_symmetric_tensor(self):
        spec = "ar,br,cr->abcr"
        and_per_r = np.einsum(spec, self.P, self.P, self.P).astype(np.uint8)
        return (np.sum(and_per_r, axis=-1) & np.uint8(1)).astype(np.uint8)
    
    def add_to_factors(self, zs: Tuple[np.ndarray], indexes: Tuple[np.ndarray]):
        if not np.all(self.T == self.to_symmetric_tensor()):
            raise RuntimeError("adding non valid factors in init")
        for z, index in zip(zs, indexes):
            self.P[:,index[0]] ^= z
        
        if not np.all(self.T == self.to_symmetric_tensor()):
            print(indexes)
            raise RuntimeError("adding non valid factors")
        
    def add_factors(self, ys: np.ndarray):
        oshape = self.P.shape
        if oshape[0] != oshape[1]:
            raise ValueError("asdf")
        new_P = np.zeros((oshape[0], oshape[1] + ys[1])).astype(np.uint8)
        new_P[:,:oshape[1]] = self.P
        new_P[:,oshape[1]:] = ys
        print("hello")

    def destroy_duplicate_columns(self):
        indexes = set()
        for i in range(self.P.shape[1]):
            if np.all(self.P[:,i] == 0):
                indexes.add(i)
            if i in indexes:
                continue
            for j in range(i + 1, self.P.shape[1]):
               if np.all(self.P[:,i] == self.P[:,j]):
                    indexes.add(i)
                    indexes.add(j)
                    break
        oshape = self.P.shape
        new_P = np.zeros((oshape[0], oshape[1]  - len(indexes))).astype(np.uint8)
        j = 0
        for i in range(self.P.shape[1]):
            if i not in indexes:
                new_P[:,j] = self.P[:,i]
                j += 1
        self.P = new_P
        if not np.all(self.T == self.to_symmetric_tensor()):
            raise RuntimeError("wrong destruction")