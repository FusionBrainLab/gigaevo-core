r"""H2 baseline: greedy interference-based deletion (general $L$).

Deactivates antennas one by one. At each step it removes the still-active antenna
whose removal leaves the smallest residual off-diagonal energy
$\sum_{i \ne j} \left|\sum_{k \in \text{active}} v_{ki}^* v_{kj}\right|^2$ of the
active Gram $V^{H} V$, preserving orthogonality of the effective channel. At
$L=2$ this is the residual $\left|\sum_{k} v_{k1}^* v_{k2}\right|$.
"""

import numpy as np


class Solver:
    def solve(self, V, K, sigma):
        n, _ = V.shape
        if K >= n:
            return np.ones(n, dtype=bool)
        Vc = V.conj()
        a2 = V.real**2 + V.imag**2  # |v_ni|^2
        term3 = a2.sum(1) ** 2 - (a2**2).sum(1)  # intrinsic, constant
        gram = Vc.T @ V
        term2 = np.einsum("ni,ni->n", V @ gram, Vc).real - a2 @ a2.sum(0)
        removed = np.zeros(n, dtype=bool)
        for _ in range(n - K):
            energy = term3 - 2.0 * term2  # off-diag energy if antenna n is dropped
            energy[removed] = np.inf
            j = int(np.argmin(energy))
            pj = Vc @ V[j]
            term2 -= pj.real**2 + pj.imag**2 - a2 @ a2[j]
            removed[j] = True
        return ~removed


def entrypoint():
    return Solver()
