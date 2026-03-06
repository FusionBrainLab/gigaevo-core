import numpy as np


def entrypoint() -> np.ndarray:
    n = 14
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5)) 
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2 
        radius = np.sqrt(1 - y * y) 
        theta = golden_angle * i  
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([0.5 + 0.4 * x, 0.5 + 0.4 * y, 0.5 + 0.4 * z])
    return np.array(points, dtype=np.float32)

