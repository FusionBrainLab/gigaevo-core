import numpy as np
from helper import get_residual_num, reconstruct_from_multi_binary_factors



def validate(
    payload: tuple[list[object], list[dict[str, np.ndarray]]],
    W_EXACT: float = 20.0,      # bonus for residual==0
    W_UNDER_SOTA: float = 100.0,   # extra bonus if rank < sota
    W_AT_SOTA: float = 50.0,       # blonus if rank == sota
) -> dict[str, float]:
    context, result = payload
    score = 0
    if (len(result) == 0 or len(context) == 0):
        return {"fitness": 11, "is_valid": 0}
    for con, res in zip(context, result):
        T_rec = reconstruct_from_multi_binary_factors(res["factors"])
        residual = get_residual_num(con.tensor, T_rec)
        rank = res["factors"].shape[-1]
        score += 5*(1 - residual/T_rec.size)
        if residual == 0:
            score += W_EXACT
            if rank == con.sota_rank:
                score += W_AT_SOTA
            if rank < con.sota_rank:
                score += W_UNDER_SOTA
    if (score < 0):
        return {"fitness": 50, "is_valid": 1}
    return {"fitness": score, "is_valid": 1}
