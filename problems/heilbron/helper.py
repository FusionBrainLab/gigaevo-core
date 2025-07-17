import numpy as np

def get_unit_triangle() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit_area_side = np.sqrt(4 / np.sqrt(3))  # scale for unit area
    height = np.sqrt(3) / 2 * unit_area_side
    A = np.array([0, 0])
    B = np.array([unit_area_side, 0])
    C = np.array([unit_area_side / 2, height])
    return A, B, C