from typing import Iterable
from pyswarm import pso
import numpy as np
from helper import get_matrix, BaseEvaluator, Matrix, ExplorationScore, FinalizationScore, get_matrix
import random
import cma

np.random.seed(42)
random.seed(40)

def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))

def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0/beta) * np.log(np.exp(-beta*(xs - m)).sum()))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Evaluator(BaseEvaluator):
    seeds = [random.randint(1, 10000) for _ in range(1)]

    def policy_mapping(self):
        ranks = [520, 460, 0]
        eweigts = [self.map_par(_w_tanh, 0) for i in range(5)]
        fweigts = [self.map_par(_w_tanh, 0) for i in range(6)]
        red_diffs = [0, self.map_par(_w_tanh, 0), self.map_par(_w_tanh, 0)] # adding correlations to parameters
        weights_diffs = [0, self.map_par(_w_tanh, 0), self.map_par(_w_tanh, 0)]# adding correlations to parameters
        tohpe_mult = [1, self.map_par(_w_tanh, 0), 0] # tohpe at the end always zero size so do not wast computation on it

        w_pool = [ExplorationScore(
            weights=eweigts,
            pow=1
        ) for r, w in zip(red_diffs, weights_diffs)]
        
        w_final = [FinalizationScore(
            weights=[fweigts[0] + r,
            fweigts[1],
            fweigts[2] + r,
            fweigts[3] + w,
            fweigts[4] + w,
            t*fweigts[5]],
            centers=[0, 0.1, 0.5, 0.5, 0.5,0],
            pow=2
        ) for r, w, t in zip(red_diffs, weights_diffs, tohpe_mult)]
        
        self.set_pool_scores(ranks, w_pool)
        self.set_final_scores(ranks, w_final)
        self.set_min_pool_size(2)
        self.set_min_z_to_research(10 + self.map_par(sigmoid, 0) * 200) 
        self.set_temperature(0.2)
        
        num_samples_vals = [8, 6, 4] 
        self.set_num_samples(ranks, num_samples_vals)
        self.set_max_pool_size((460, 0), (20, 100))
        self.set_max_tohpe(6)
        
        try_only_tohpe_vals = [0, 1, 1]  
        self.set_try_only_tohpe(ranks, try_only_tohpe_vals)
        
        self.set_max_reduction(10)
        self.set_min_reduction(2)
        self.set_max_from_single_ns(2)
        self.set_tohpe_num_best(10)
        self.set_gen_part(0.8)
        self.set_todd_width(2)  
        self.set_beamsearch_width((440,0), (2,6))  

    def __call__(self, params: Iterable): 
        tcounts = self.run(params, self.seeds)
        bestish = softmin(tcounts, beta=6.0)
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return bestish + 0.01 * spread

def run_cma(fun, num_eval: int = 5, initial_sigma: float = 0.5) -> np.ndarray:
    x0 = fun.extract_active()
    x0 = np.zeros_like(x0)
    bounds = [-2.0, 2.0]
    def objective_function(x):
        if x.ndim == 1:
            return float(fun(x))
        else:
            return np.array([fun(xi) for xi in x])
    
    options = {
        'maxfevals': num_eval,
        'popsize' : 9,
        'bounds': bounds,
        'verbose': 0,
        'tolfun': 1e-4,
        'tolx': 1e-4,
    }
    
    xopt, es = cma.fmin2(
        objective_function,
        x0,
        initial_sigma,  
        options=options,
        restarts=0,
        bipop=False
    )
    
    return np.array(xopt)

def entrypoint(mat):
    fun = Evaluator(mat=mat, max_depth=200)
    
    xopt = run_cma(fun, 30)
    x_active = fun.set_up_new_init(0, rank_thr=460, xopt=xopt)
    if x_active is not None and len(x_active) > 0:
        xopt = run_cma(fun, 50)
        fun(xopt)
    
    return fun.get_best()