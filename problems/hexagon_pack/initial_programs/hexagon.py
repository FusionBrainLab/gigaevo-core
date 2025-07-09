import numpy as np
import random
from helper import compute_outer_hex_side_length, get_hexagon_vertices, check_hexagon_overlap_many, check_hexagon_overlap_two

def entrypoint() -> tuple[np.ndarray, np.ndarray]:

    centers = []
    angles = []

    radius_step = 2.2
    angle_offset = 0.0
    placed = 0
    max_rings = 5

    for ring in range(max_rings):
        if placed >= 11:
            break
        ring_radius = radius_step * ring
        if ring == 0:
            centers.append([0.0, 0.0])
            angles.append(random.uniform(0, np.pi / 3))
            placed += 1
        else:
            num_in_ring = 6 * ring
            for i in range(num_in_ring):
                if placed >= 11:
                    break
                theta = 2 * np.pi * i / num_in_ring + angle_offset
                centers.append([ring_radius * np.cos(theta), ring_radius * np.sin(theta)])
                angles.append(random.uniform(0, np.pi / 3))
                placed += 1

    centers = np.array(centers)
    angles = np.array(angles)

    return centers, angles
