def entrypoint():
    """Nelder-Mead simplex — gradient-free, easily trapped by local minima."""

    def optimizer(f, bounds, budget):
        dim = len(bounds)
        evals = [0]

        def eval_f(x):
            if evals[0] >= budget:
                return float("inf")
            evals[0] += 1
            return f(x)

        # Initialize simplex
        x0 = [(lo + hi) / 2 for lo, hi in bounds]
        simplex = [x0[:]]
        for i in range(dim):
            v = x0[:]
            v[i] += (bounds[i][1] - bounds[i][0]) * 0.1
            simplex.append(v)

        vals = [eval_f(v) for v in simplex]

        while evals[0] < budget:
            order = sorted(range(dim + 1), key=lambda i: vals[i])
            simplex = [simplex[i] for i in order]
            vals = [vals[i] for i in order]

            # Centroid (exclude worst)
            centroid = [
                sum(simplex[j][i] for j in range(dim)) / dim for i in range(dim)
            ]

            # Reflect
            xr = [2 * centroid[i] - simplex[-1][i] for i in range(dim)]
            xr = [max(bounds[i][0], min(bounds[i][1], xr[i])) for i in range(dim)]
            fr = eval_f(xr)

            if fr < vals[0]:
                # Expand
                xe = [3 * centroid[i] - 2 * simplex[-1][i] for i in range(dim)]
                xe = [max(bounds[i][0], min(bounds[i][1], xe[i])) for i in range(dim)]
                fe = eval_f(xe)
                if fe < fr:
                    simplex[-1], vals[-1] = xe, fe
                else:
                    simplex[-1], vals[-1] = xr, fr
            elif fr < vals[-2]:
                simplex[-1], vals[-1] = xr, fr
            else:
                # Contract
                xc = [0.5 * (centroid[i] + simplex[-1][i]) for i in range(dim)]
                fc = eval_f(xc)
                if fc < vals[-1]:
                    simplex[-1], vals[-1] = xc, fc
                else:
                    # Shrink
                    for j in range(1, dim + 1):
                        simplex[j] = [
                            0.5 * (simplex[0][i] + simplex[j][i]) for i in range(dim)
                        ]
                        vals[j] = eval_f(simplex[j])

        best_idx = min(range(len(vals)), key=lambda i: vals[i])
        return simplex[best_idx]

    return optimizer
