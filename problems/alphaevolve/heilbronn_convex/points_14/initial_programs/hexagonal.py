import numpy as np


def entrypoint():
    np.random.seed(123)
    points = []

    center_x, center_y = 0.5, 0.5
    points.append([center_x, center_y])

    radius1 = 0.25
    for i in range(6):
        angle = i * np.pi / 3 + np.random.uniform(-0.05, 0.05)
        r = radius1 + np.random.uniform(-0.02, 0.02)
        x = center_x + r * np.cos(angle)
        y = center_y + r * np.sin(angle)
        points.append([x, y])

    radius2 = 0.4
    for i in range(7):
        angle = i * 2 * np.pi / 7 + np.pi / 7 + np.random.uniform(-0.05, 0.05)
        r = radius2 + np.random.uniform(-0.02, 0.02)
        x = center_x + r * np.cos(angle)
        y = center_y + r * np.sin(angle)
        points.append([x, y])

    points = np.array(points[:14], dtype=np.float32)

    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0) + 1e-10)
    points = points * 0.8 + 0.1

    return points.astype(np.float32)

