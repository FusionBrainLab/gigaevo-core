"""Five-dimensional cut space of the six-vertex wheel graph."""

import numpy as np


def entrypoint() -> np.ndarray:
    # Vertex 0 is the hub; vertices 1,...,5 form a cycle.  Delete the final
    # incidence column to obtain a full-rank representation of the cut space.
    edges = [(0, vertex) for vertex in range(1, 6)]
    edges += [(vertex, vertex % 5 + 1) for vertex in range(1, 6)]

    incidence = np.zeros((10, 5), dtype=np.float64)
    for edge_index, (source, target) in enumerate(edges):
        if source < 5:
            incidence[edge_index, source] += 1.0
        if target < 5:
            incidence[edge_index, target] -= 1.0
    return incidence
