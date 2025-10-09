from typing import List, Dict, Any
import torch
from helper import reconstruct_from_single_binary_factor, get_residual_num, Data  # assuming torch versions

def entrypoint(context: List[Data]) -> List[Dict[str, Any]]:
    def _sample_rank1(rng: torch.Generator, d: int) -> torch.Tensor:
        v = (torch.rand(d, generator=rng) < 0.5).to(torch.uint8)
        if v.sum() == 0:  # ensure non-zero
            v[torch.randint(0, d, (1,), generator=rng)] = 1
        return v

    def _choose_best(rng: torch.Generator, residual: torch.Tensor, samples: int):
        best_score, best = float("inf"), None
        d = int(residual.shape[0])
        for _ in range(samples):
            cand = _sample_rank1(rng, d)                      # (d,) uint8
            rank1 = reconstruct_from_single_binary_factor(cand)  # (d,d,d) uint8
            sc = int(get_residual_num(residual, rank1))
            if sc < best_score:
                best_score, best = sc, cand
        return best, best_score

    def search_min_rank(T: torch.Tensor, samples=128, max_rank=64, tol=1e-6, seed=0) -> Dict[str, Any]:
        R = torch.as_tensor(T, dtype=torch.uint8).contiguous()
        rng = torch.Generator().manual_seed(seed)
        factors = []
        cur = get_residual_num(R)

        for _ in range(max_rank):
            cand, sc = _choose_best(rng, R, samples)
            if cand is None or cur < tol: break
            R = (R ^ reconstruct_from_single_binary_factor(cand)).to(torch.uint8)  # GF(2) update
            factors.append(cand)
            cur = get_residual_num(R)
            if cur < tol: break

        F = torch.stack(factors, dim=1).to(torch.uint8) if factors else torch.zeros((R.shape[0],0), dtype=torch.uint8)
        return {"rank": F.shape[1], "residual": get_residual_num(R), "factors": F}

    out = []
    for i, item in enumerate(context):
        T = item.tensor
        hint = int(getattr(item, "sota_rank", 8))
        max_r = int(min(2 * hint, 256))
        out.append(search_min_rank(T=T, samples=20, max_rank=max_r, tol=1e-6, seed=i+1))
    return out
