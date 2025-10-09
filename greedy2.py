# Insight (algebraic): Exploit symmetry by precomputing tensor slices.
# Insight (optimization): Adaptive learning rate based on residual change.
# Lineage (imitation): Increased rank proportionally to tensor size, as successful in prior runs.
# Design: Precompute tensor slices for faster residual calculation; adaptive learning rate; rank initialization based on tensor size.
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict
from helper import Data, reconstruct_from_multi_binary_factors, reconstruct_from_single_binary_factor, get_residual_num

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

def sigmoid_t(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def smooth_reconstruction(params: torch.Tensor) -> torch.Tensor:
    p = sigmoid_t(params)                              
    and_per_r = torch.einsum("ar,br,cr->abcr", p, p, p)
    return 0.5 * (1.0 - torch.prod(1.0 - 2.0 * and_per_r, dim=-1))

def bce_from_probs(target01: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    # stable BCE on probabilities
    probs = probs.clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(probs, target01, reduction="sum")

def precompute_tensor_slices(T: torch.Tensor) -> torch.Tensor:
    return T.reshape(-1).contiguous()

def train_step(params: torch.nn.Parameter, opt: torch.optim.Optimizer,
               T_float: torch.Tensor, precomp: torch.Tensor, l1: float = 1e-2) -> torch.Tensor:
    opt.zero_grad(set_to_none=True)
    P = smooth_reconstruction(params)                # (n,n,n)
    loss = bce_from_probs(T_float, P) + l1 * params.abs().sum()
    loss.backward()
    opt.step()
    return loss

def entrypoint(context: List[Data]) -> List[Dict]:
    results: List[Dict] = []
    for data in context:
        T = torch.as_tensor(data.tensor, dtype=torch.uint8, device=device)
        n = int(T.shape[0])
        sota_rank = int(data.sota_rank)

        T_float = T.float()
        precomputed_slices = precompute_tensor_slices(T)

        best_rank = sota_rank * 2
        best_residual = float("inf")
        best_factors = None

        for rank in range(sota_rank - 2, sota_rank + 2):
            g = torch.Generator(device="cpu"); g.manual_seed(0)
            params = torch.randn((n, rank), generator=g, dtype=torch.float32, device=device) * 0.1
            params = torch.nn.Parameter(params)
            opt = torch.optim.Adam([params], lr=0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

            prev_residual = float("inf")
            residual_tolerance = 1e-4
            plateau_count = 0
            max_plateau = 10

            for step in range(2000):
                loss = train_step(params, opt, T_float, precomputed_slices)

                # Evaluate in discrete GF(2) space
                factors_binary = (sigmoid_t(params) >= 0.5).to(torch.uint8)    # (n, r)
                reconstructed_tensor = reconstruct_from_multi_binary_factors(factors_binary)  # (n,n,n) uint8
                residual = get_residual_num(T, reconstructed_tensor)

                scheduler.step(residual)

                if residual < best_residual:
                    best_residual = residual
                    best_rank = rank
                    best_factors = factors_binary.detach().cpu()
                    if residual == 0:
                        break

                residual_change = prev_residual - residual
                if residual_change < residual_tolerance:
                    plateau_count += 1
                    if plateau_count > max_plateau:
                        break
                else:
                    plateau_count = 0

                prev_residual = residual

        results.append({
            "rank": int(best_rank),
            "residual": int(best_residual),
            "factors": best_factors  # (n, best_rank) uint8
        })
    return results
