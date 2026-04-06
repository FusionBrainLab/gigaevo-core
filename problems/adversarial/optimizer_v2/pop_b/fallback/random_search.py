def entrypoint():
    """Pure random search — simplest possible optimizer."""
    import random

    def optimizer(f, bounds, budget):
        best_x = None
        best_val = float("inf")
        for _ in range(budget):
            x = [random.uniform(lo, hi) for lo, hi in bounds]
            val = f(x)
            if val < best_val:
                best_val = val
                best_x = x[:]
        return best_x

    return optimizer
