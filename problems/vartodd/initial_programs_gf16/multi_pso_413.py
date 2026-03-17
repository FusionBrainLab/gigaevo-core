from typing import Iterable
import cma
import nlopt
import pyswarms as ps
import numpy as np
from helper import get_matrix, BaseEvaluator, Matrix, ExplorationScore, FinalizationScore, get_matrix
import random

np.random.seed(42)
random.seed(42)

def _w_tanh(z: float, scale: float = 4.5, sharp: float = 1.2) -> float:
    return float(scale * np.tanh(z / sharp))

def _sig(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(4)]

    def policy_mapping(self):
        ranks = [440, 415, 0]
        
        w_pool = []
        for r in ranks:
            weights = [self.map_par(_w_tanh, r) for _ in range(5)]
            # Bias wred to be negative at low ranks to avoid local minima
            if r < 415:
                weights[0] = -abs(weights[0]) - 0.5
            w_pool.append(ExplorationScore(weights, pow=1))

        w_final = []
        for r in ranks:
            weights = [self.map_par(_w_tanh, r) for _ in range(6)]
            centers = [self.map_par(_sig, r) for _ in range(6)]
            w_final.append(FinalizationScore(weights, centers, pow=2))

        self.set_pool_scores(ranks, w_pool)
        self.set_final_scores(ranks, w_final)

        self.set_min_pool_size(2)
        self.set_max_pool_size(ranks, [32, 48, 64])
        self.set_min_z_to_research(ranks, [300, 200, 100])
        self.set_temperature(0.15)

        self.set_num_samples(ranks, [16, 12, 8])

        self.set_max_tohpe(ranks, [60, 40, 20])
        self.set_try_only_tohpe(ranks, [1, 0, 0]) 

        self.set_max_reduction(25)
        self.set_min_reduction(1)
        self.set_max_from_single_ns(4)
        self.set_tohpe_num_best(ranks, [8, 4, 2])
        self.set_gen_part(0.65)
        
        self.set_todd_width(ranks, [2, 4, 4])
        self.set_beamsearch_width(ranks, [2, 6, 4])

    def __call__(self, params: Iterable):
        tcounts = self.run(params, self.seeds)
        if not tcounts: return 1000.0
        return float(np.min(tcounts)) + 0.015 * float(np.std(tcounts))

def run_pso(fun, lb, ub, swarmsize=8, maxiter=5, options={ 'c1': 0.6, 'c2': 0.6, 'w': 0.7 }) -> Evaluator:
    x = fun.extract_active()
    n_params = len(x)
    def objective_function(positions):
        return np.array([fun(pos) for pos in positions])
    
    bounds = (np.array(lb), np.array(ub))

    optimizer = ps.single.GlobalBestPSO(
        n_particles=swarmsize, 
        dimensions=n_params,
        options=options,
        bounds=bounds
    )

    best_cost, best_position = optimizer.optimize(
        objective_function, 
        iters=maxiter,
        verbose=False
    )
    return best_position
        

def entrypoint(mat):
    fun = Evaluator(mat=mat, max_depth=250)
    
    x_active = fun.extract_active()
    if not x_active: return fun.get_best()
    lb = [-2.5] * len(x_active)
    ub = [2.5] * len(x_active)
    xopt = run_pso(fun, lb, ub, swarmsize=4, maxiter=5)
    
    x_active = fun.set_up_new_init(0, rank_thr=430, xopt=xopt)
    if x_active is not None:
        lb2 = [-1.5] * len(x_active)
        ub2 = [1.5] * len(x_active)
        xopt = run_pso(fun, lb2, ub2, swarmsize=6, maxiter=6)
    
    return fun.get_best()