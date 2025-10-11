import numpy as np
import jax.numpy as jnp
from helper import get_residual_num, reconstruct_from_multi_binary_factors, Data


def validate(
    payload: tuple[list[Data], list[jnp.ndarray]],
    W_EXACT: float = 3.0,      # bonus for residual==0
    W_UNDER_SOTA: float = 20.0,   # extra bonus if rank < sota
    W_AT_SOTA: float = 3.0,       # blonus if rank == sota
) -> dict[str, float]:
    context, result = payload
    score = 0
    exact_num = 0
    under_sota_num = 0 
    at_sota_num = 0
    for con, res in zip(context, result):
        T_rec = reconstruct_from_multi_binary_factors(res)
        residual = get_residual_num(con.tensor, T_rec)
        rank = res.shape[-1] 
        score -= residual/T_rec.size * 5
        if residual == 0:
            exact_num = +1
            score += W_EXACT
            if rank == con.sota_rank:
                at_sota_num += 1
                score += W_AT_SOTA
            if rank < con.sota_rank:
                under_sota_num += 1
                score += W_UNDER_SOTA
    return {"fitness": score, "exact_num": exact_num, "under_sota": under_sota_num, "at_sota": at_sota_num, "is_valid": 1}