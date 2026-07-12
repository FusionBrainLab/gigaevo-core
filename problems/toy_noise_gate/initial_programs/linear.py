import numpy as np


def entrypoint():
    x = np.linspace(-2.0, 2.0, 64)
    return [float(v) for v in 0.4 * x]
