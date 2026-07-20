"""Dense near-equality perturbation of the sparse series-parallel seed."""

import numpy as np


def _theta_basis() -> np.ndarray:
    edges = [(0, 1), (0, 1)]
    weights = [1.0, 1.0]
    for middle_vertex in range(2, 6):
        edges.extend([(0, middle_vertex), (middle_vertex, 1)])
        weights.extend([16.0 / 9.0, 16.0 / 9.0])

    matrix = np.zeros((10, 5), dtype=np.float64)
    for edge_index, ((source, target), weight) in enumerate(zip(edges, weights)):
        scale = np.sqrt(weight)
        if source < 5:
            matrix[edge_index, source] += scale
        if target < 5:
            matrix[edge_index, target] -= scale
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return basis


def entrypoint() -> np.ndarray:
    basis = _theta_basis()
    rng = np.random.default_rng(2026)
    perturbation = rng.standard_normal(basis.shape)

    # Keep only the horizontal (Grassmann tangent) component: an in-subspace
    # component would merely change the arbitrary five-column basis.
    perturbation -= basis @ (basis.T @ perturbation)
    perturbed, _ = np.linalg.qr(
        basis + 0.003 * perturbation,
        mode="reduced",
    )
    return perturbed
