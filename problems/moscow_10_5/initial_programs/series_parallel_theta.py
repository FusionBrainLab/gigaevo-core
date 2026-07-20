"""Sharp Phi=1 series-parallel construction (the equality regression seed)."""

import numpy as np


def entrypoint() -> np.ndarray:
    # Six-vertex graph: two direct terminal edges in parallel with four
    # two-edge terminal paths.  Direct weights are 1; path-edge weights 16/9.
    edges = [(0, 1), (0, 1)]
    weights = [1.0, 1.0]
    for middle_vertex in range(2, 6):
        edges.extend([(0, middle_vertex), (middle_vertex, 1)])
        weights.extend([16.0 / 9.0, 16.0 / 9.0])

    weighted_incidence = np.zeros((10, 5), dtype=np.float64)
    for edge_index, ((source, target), weight) in enumerate(zip(edges, weights)):
        scale = np.sqrt(weight)
        if source < 5:
            weighted_incidence[edge_index, source] += scale
        if target < 5:
            weighted_incidence[edge_index, target] -= scale
    return weighted_incidence
