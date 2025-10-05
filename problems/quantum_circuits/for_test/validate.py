import numpy as np
from helper import get_residual_num, reconstruct_from_multi_binary_factors

def validate(
    payload: tuple[list[object], list[dict[str, np.ndarray]]],
    W_EXACT: float = 10.0,      # bonus for residual==0
    W_UNDER_SOTA: float = 500.0,   # extra bonus if rank < sota
    W_AT_SOTA: float = 50.0,       # bonus if rank == sota
) -> dict[str, float]:
    context, result = payload
    score = 0
    for con, res in zip(context, result):
        T_rec = reconstruct_from_multi_binary_factors(res["factors"])
        residual = get_residual_num(con.tensor, T_rec)
        rank = res["factors"].shape[-1]
        if residual == 0:
            score += W_EXACT
            if rank == con.m:
                score += W_AT_SOTA
            if rank < con.m:
                score += W_UNDER_SOTA
        else:
            score -= res["residual"]
        score -= res["steps"]/ 100_000
    return {"fitness": score, "is_valid": 1}
