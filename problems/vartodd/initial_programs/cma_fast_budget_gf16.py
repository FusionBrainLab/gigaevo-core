from typing import Iterable
import numpy as np
import random
import cma
from helper import BaseEvaluator, Matrix, ExplorationScore, FinalizationScore

np.random.seed(42)
random.seed(40)

def _w_tanh(z: float, scale: float = 2.8, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(1)]

    def policy_mapping(self):
        ranks = [450, 0]

        wred = self.map_par(_w_tanh, 0)
        wdim = self.map_par(_w_tanh, 0)
        wbucket = self.map_par(_w_tanh, 0)
        fwred = self.map_par(_w_tanh, 0)
        fwdim = self.map_par(_w_tanh, 0)
        fwbucket = self.map_par(_w_tanh, 0)
        budget = self.map_par(sigmoid, 0)

        w_pool = [ExplorationScore(weights=[wred, wdim, wbucket, 0.0, 0.0], pow=1) for _ in ranks]
        w_final = [FinalizationScore(
            weights=[fwred, fwdim, fwbucket, 0.0, 0.0, 0.0],
            centers=[1, 0, 0, 0, 0, 0],
            pow=2
        ) for _ in ranks]

        self.set_pool_scores(ranks, w_pool)
        self.set_final_scores(ranks, w_final)

        beam = int(1 + 2 * budget)
        todd_w = int(1 + 2 * budget)
        max_pool = int(12 + 18 * budget)
        num_samples = int(5 + 8 * (1.0 - budget))
        min_z = 8 + 120 * budget
        max_tohpe = int(2 + 2 * budget)
        gen_part = 0.25 + 0.5 * budget

        self.set_min_pool_size(2)
        self.set_min_z_to_research(min_z)
        self.set_temperature(0.5)
        self.set_num_samples(ranks, [num_samples, max(4, num_samples - 2)])
        self.set_max_pool_size(max_pool)
        self.set_max_tohpe(max_tohpe)
        self.set_try_only_tohpe(ranks, [1, 1])
        self.set_max_reduction(8)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(2)
        self.set_tohpe_num_best(6)
        self.set_gen_part(gen_part)
        self.set_todd_width(todd_w)
        self.set_beamsearch_width(beam)

    def __call__(self, params: Iterable):
        early = self.run(params, self.seeds[:1])
        if self.best_rank < 100000 and early[0] > self.best_rank + 25:
            return float(early[0] + 8.0)

        rest = self.run(params, self.seeds[1:]) if len(self.seeds) > 1 else []
        tcounts = early + rest
        q = float(np.quantile(tcounts, 0.7))
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return q + 0.02 * spread


def run_cma(fun: Evaluator, maxfevals: int = 50, sigma: float = 0.35) -> np.ndarray:
    x0 = fun.extract_active()
    x0 = np.zeros_like(x0)
    bounds = [-1.5, 1.5]

    def objective_function(x):
        if x.ndim == 1:
            return float(fun(x))
        return np.array([fun(xi) for xi in x])

    options = {
        "maxfevals": maxfevals,
        "popsize": 6,
        "bounds": bounds,
        "verbose": 0,
        "tolfun": 1e-4,
        "tolx": 1e-4,
        "seed": 42,
    }

    xopt, es = cma.fmin2(
        objective_function,
        x0,
        sigma,
        options=options,
        restarts=0,
        bipop=False,
    )

    return np.array(xopt)


def entrypoint():
    fun = Evaluator(path_name="init", init_rank_thr=0)

    xopt = run_cma(fun, maxfevals=80, sigma=0.4)
    fun(xopt)

    best_rank = fun.best_rank
    x_active = fun.set_up_new_init(0, rank_thr=int(best_rank + 25), xopt=xopt)
    if x_active is not None and len(x_active) > 0:
        xopt = run_cma(fun, maxfevals=60, sigma=0.3)
        fun(xopt)

    return fun.get_best()
