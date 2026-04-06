"""Cross-play evaluator: optimizer vs opponent landscapes.

Called by CallValidatorFunction as evaluate(opponent_results, program_output).
All results are pre-computed by the pipeline -- no exec() here.

opponent_results: list of (objective, bounds, optimum, budget) tuples
    from FetchOpponentResultsStage (each opponent's entrypoint() result).
program_output: callable optimizer(f, bounds, budget) -> x_best
    from CallProgramFunction (this program's entrypoint() result).
"""


def evaluate(opponent_results, program_output):
    import math

    optimize = program_output
    if not callable(optimize):
        return {"fitness": 0.0, "is_valid": 0.0, "n_opponents": 0.0}

    scores = []
    for landscape in opponent_results:
        try:
            if not isinstance(landscape, tuple) or len(landscape) != 4:
                continue
            objective, bounds, optimum, budget = landscape
            dim = len(optimum)
            x_best = optimize(objective, bounds, int(budget))
            if x_best is None or len(x_best) != dim:
                scores.append(0.0)
                continue
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(x_best, optimum)))
            max_dist = math.sqrt(dim) * 10  # diagonal of [-5,5]^dim
            scores.append(1.0 - min(dist / max_dist, 1.0))
        except Exception:
            scores.append(0.0)

    fitness = sum(scores) / len(scores) if scores else 0.0
    return {
        "fitness": fitness,
        "is_valid": 1.0,
        "n_opponents": float(len(scores)),
    }
