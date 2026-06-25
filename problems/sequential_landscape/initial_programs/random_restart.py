def entrypoint():
    """Naive seed: numpy random sampling then greedy local refinement.

    Has no notion of the basin chain, so it stalls in an early shallow basin
    on the harder ladder instances.
    """
    import numpy as np

    class Optimizer:
        def optimize(self, f, bounds, budget):
            rng = np.random.default_rng(0)
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            dim = len(bounds)

            best_x = None
            best_v = np.inf
            used = 0

            explore = int(budget * 0.6)
            while used < explore:
                x = lo + (hi - lo) * rng.random(dim)
                v = f(x)
                used += 1
                if v < best_v:
                    best_v, best_x = v, x

            step = 0.1
            while used < budget:
                cand = np.clip(best_x + rng.normal(0, step * (hi - lo)), lo, hi)
                v = f(cand)
                used += 1
                if v < best_v:
                    best_v, best_x = v, cand
                else:
                    step *= 0.97

            return best_x

    return Optimizer()
