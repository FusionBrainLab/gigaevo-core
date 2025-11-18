import numpy as np


def entrypoint():
    np.random.seed(42)
    grid_size = 4
    x_coords = np.linspace(0.1, 0.9, grid_size)
    y_coords = np.linspace(0.1, 0.9, grid_size)

    grid_points = []
    for x in x_coords:
        for y in y_coords:
            grid_points.append([x, y])

    selected_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15]
    points = np.array([grid_points[i] for i in selected_indices[:13]], dtype=np.float32)

    perturbation = np.random.uniform(-0.02, 0.02, (13, 2))
    points += perturbation

    return points
