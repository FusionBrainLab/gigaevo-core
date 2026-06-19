"""Antenna Selection validator.

The evolved program returns a ``Solver`` instance (via ``entrypoint()``). On every
call this validator draws FRESH random channel matrices (new entropy) across an
``(N, L)`` grid — ``N ∈ {1000, 1500, 2000, 2500, 3000, 4000}``, ``L = 2..10``,
``P = 1``, ``sigma = 1`` (Task 3) — asks the solver for an active-antenna mask on
each, and scores the General objective ``det(V_eq V_eqᴴ + σI)`` against the better
of two heuristics (H1, H2). Both the objective and H2 are L-general.

Because the matrices are redrawn every call, an evolved solver cannot memorize or
hardcode specific instances — fitness is a low-variance generalization estimate
over the 108 instances, not a fixed number. Never prints. ``validate`` returns
``(metrics, feedback_text)`` where the feedback is a per-(N, L) capacity-gain summary
shown to the mutator; an unusable solver raises ``InvalidProgramError``.
"""

from __future__ import annotations

import numpy as np

N_VALUES = (1000, 1500, 2000, 2500, 3000, 4000)
L_VALUES = (2, 3, 4, 5, 6, 7, 8, 9, 10)
SEEDS_PER_CELL = 2  # fresh instances per (N, L) cell -> 6 * 9 * 2 = 108 instances
SIGMA = 1.0
P = 1.0
LN2 = float(np.log(2.0))


class InvalidProgramError(ValueError):
    """The evolved solver is unusable on some instance: its ``solve()`` raised, it
    returned a malformed mask, or its active set is out of range. Raising (rather
    than returning a sentinel) lets the validator stage fail cleanly — the metrics
    stage records sentinel metrics (``is_valid=0``) by design, and this message
    reaches the mutator as failure feedback."""


def generate_V(n: int, ell: int, rng) -> np.ndarray:
    """Reference channel generator (P = 1).

    ``rng`` accepts an int seed (a reproducible instance, used by the offline
    evaluators), a shared ``np.random.Generator`` (fresh advancing draws, used by
    ``validate``), or ``None`` (fresh). Columns are unit-norm and the strongest
    antenna's power is normalised to 1.
    """
    rng = np.random.default_rng(rng)
    V = rng.normal(size=(n, ell)) + 1j * rng.normal(size=(n, ell))
    V /= np.linalg.norm(V, axis=0)  # unit columns
    V /= np.max(np.linalg.norm(V, axis=1))  # max per-antenna power = 1
    return V


def _objective_logdet(V: np.ndarray, mask: np.ndarray, sigma: float, p: float) -> float:
    """General objective on the L×L effective channel: log det(V_eq V_eqᴴ + σI)."""
    kept = np.flatnonzero(mask)
    s = np.sum(np.abs(V) ** 2, axis=1)  # per-antenna power s_n
    z = np.sqrt(
        p / np.max(s)
    )  # constraint pinned to the GLOBAL strongest antenna (generate_V → 1): fixed across
    #   masks so the solver cannot inflate the objective by dropping high-power antennas
    W = np.zeros_like(V)
    W[kept] = z * V[kept]
    Veq = V.conj().T @ W  # (L, L) Hermitian PSD
    gram = Veq @ Veq.conj().T + sigma * np.eye(Veq.shape[0])
    _, logdet = np.linalg.slogdet(gram)
    return float(logdet)


def _h1_mask(V: np.ndarray, k: int) -> np.ndarray:
    """H1: keep the k strongest antennas by power, turn off the weakest."""
    keep = np.argsort(np.sum(np.abs(V) ** 2, axis=1))[::-1][:k]
    mask = np.zeros(V.shape[0], dtype=bool)
    mask[keep] = True
    return mask


def _h2_mask(V: np.ndarray, k: int) -> np.ndarray:
    """H2 (general L): greedily delete the antenna that most reduces the residual
    off-diagonal energy ``||offdiag(Vᴴ V)||_F²`` of the active set. At L=2 this is
    exactly the PDF rule (min residual ``|Σ v*_n1 v_n2|``).

    Vectorized to O(N²L): the off-diagonal energy after dropping antenna n equals
    ``E_G − 2·Re(term2_n) + term3_n`` where ``term3_n = s_n² − Σ_i|v_ni|⁴`` is
    intrinsic (constant across steps) and
    ``term2_n = Σ_{m active}(|⟨v_n,v_m⟩|² − Σ_i|v_ni|²|v_mi|²)`` is maintained by
    subtracting the deleted antenna's column contribution each step. ``E_G`` is the
    same for every candidate and drops out of the per-step argmin.
    """
    n, ell = V.shape
    if k >= n:
        return np.ones(n, dtype=bool)
    Vc = V.conj()
    a2 = V.real**2 + V.imag**2  # |v_ni|²
    term3 = a2.sum(1) ** 2 - (a2**2).sum(1)
    gram = Vc.T @ V
    term2 = np.einsum("ni,ni->n", V @ gram, Vc).real - a2 @ a2.sum(0)
    removed = np.zeros(n, dtype=bool)
    for _ in range(n - k):
        energy = term3 - 2.0 * term2
        energy[removed] = np.inf
        j = int(np.argmin(energy))
        pj = Vc @ V[j]
        term2 -= pj.real**2 + pj.imag**2 - a2 @ a2[j]
        removed[j] = True
    return ~removed


def _best_heur_logdet(V: np.ndarray, k: int) -> float:
    """Best-of-heuristics baseline for one instance: ``max(U(H1), U(H2))``.

    Recomputed per instance — matrices are fresh every call, so there is nothing
    to cache; the vectorized H2 keeps the full grid affordable (~4 s / call)."""
    return max(
        _objective_logdet(V, _h1_mask(V, k), SIGMA, P),
        _objective_logdet(V, _h2_mask(V, k), SIGMA, P),
    )


def _coerce_mask(raw, n: int) -> tuple[np.ndarray | None, str | None]:
    """Validate a solver output as a binary mask of shape (n,).

    Returns (mask, None) on success, or (None, reason) with a feedback message
    naming exactly why the output was rejected. Never raises.
    """
    try:
        arr = np.asarray(raw)
        if arr.dtype == object:
            return None, (
                f"mask is a ragged or mixed-type sequence; return a 1-D array of {n} "
                "booleans or 0/1 ints"
            )
        if np.iscomplexobj(arr):
            return None, (
                "mask is complex-valued; return a real boolean or {0,1} array "
                "(did you return the weights or V_eq by mistake?)"
            )
        flat = arr.reshape(-1)
        if flat.shape != (n,):
            return None, (
                f"mask has {flat.size} entries (shape {arr.shape}); expected a flat "
                f"array of exactly {n} (one per antenna)"
            )
        if flat.dtype == bool:
            return flat, None
        if not np.issubdtype(flat.dtype, np.number):
            return None, (
                f"mask dtype is {flat.dtype}; return booleans or 0/1 ints, "
                "not strings/objects"
            )
        if not np.all(np.isfinite(flat)):
            return None, "mask contains NaN/Inf; entries must be exactly 0 or 1"
        if not np.all((flat == 0) | (flat == 1)):
            bad = np.unique(flat[(flat != 0) & (flat != 1)])
            return None, (
                f"mask is not binary; entries must be 0 or 1 but found {bad[:5].tolist()} "
                "(return a boolean mask, not scores/probabilities)"
            )
        return flat.astype(bool), None
    except Exception as exc:  # noqa: BLE001 - last resort: never raise, always feed back
        return None, f"could not interpret solver output as a mask: {exc!r}"


def _run_grid(solver) -> list[dict]:
    """Score the solver on a fresh-random instance of every ``(N, L, seed)`` cell.

    Returns one record per instance. Raises ``InvalidProgramError`` (cell-tagged) on
    the first solver fault — a raised ``solve()``, a malformed mask, or an out-of-range
    active set. A harness/scoring fault propagates as its native exception (a genuine
    bug to surface, not the program's fault)."""
    rng = np.random.default_rng()  # fresh entropy each call -> unhackable instances
    per_instance: list[dict] = []
    for n in N_VALUES:
        k = n // 2  # keep at most N/2 active antennas (turn off 50%)
        for ell in L_VALUES:
            for instance in range(SEEDS_PER_CELL):
                cell = f"N={n} L={ell} #{instance}"
                V = generate_V(n, ell, rng)
                try:
                    raw = solver.solve(V, k, SIGMA)
                except Exception as exc:
                    raise InvalidProgramError(
                        f"solver.solve raised on {cell}: {exc!r}"
                    ) from exc

                mask, reason = _coerce_mask(raw, n)
                if mask is None:
                    raise InvalidProgramError(f"invalid mask on {cell}: {reason}")
                active = int(mask.sum())
                if not (1 <= active <= k):
                    raise InvalidProgramError(
                        f"cardinality violated on {cell}: {active} antennas active, "
                        f"must keep 1..{k} (K = N/2)"
                    )

                u_solver = _objective_logdet(V, mask, SIGMA, P)
                u_heur = _best_heur_logdet(V, k)
                per_instance.append(
                    {
                        "N": n,
                        "L": ell,
                        "instance": instance,
                        "active": active,
                        "margin": float((u_solver - u_heur) / LN2),  # bits/s/Hz
                        "logdet": u_solver,
                        "logdet_heur": u_heur,
                    }
                )
    return per_instance


def _metrics(per_instance: list[dict]) -> dict[str, float]:
    margins = [r["margin"] for r in per_instance]
    return {
        "fitness": float(np.mean(margins)),
        "win_rate": float(sum(m > 0.0 for m in margins) / len(margins)),
        "median_margin": float(np.median(margins)),
        "is_valid": 1.0,
    }


def _format_feedback(per_instance: list[dict], metrics: dict[str, float]) -> str:
    """Per-(N, L) capacity-gain summary for the mutation prompt: a mean-gain grid
    plus the weakest and already-winning regimes, so the LLM can target where the
    solver still trails the heuristics."""
    ns = sorted({r["N"] for r in per_instance})
    ells = sorted({r["L"] for r in per_instance})
    cell_mean = {
        (n, ell): float(
            np.mean(
                [r["margin"] for r in per_instance if r["N"] == n and r["L"] == ell]
            )
        )
        for n in ns
        for ell in ells
        if any(r["N"] == n and r["L"] == ell for r in per_instance)
    }
    wins = sum(r["margin"] > 0.0 for r in per_instance)
    lines = [
        "Capacity gain vs best heuristic max(H1,H2), bits/s/Hz "
        "(positive = you beat both heuristics; 0 = tie; negative = worse).",
        f"Overall mean {metrics['fitness']:+.4f} | median {metrics['median_margin']:+.4f} "
        f"| win-rate {wins}/{len(per_instance)} "
        f"({metrics['win_rate'] * 100:.0f}%) over {len(per_instance)} instances.",
        "Mean gain per (N, L) cell:",
        f"{'N/L':>6}" + "".join(f"{ell:>9}" for ell in ells),
    ]
    for n in ns:
        lines.append(
            f"{n:>6}" + "".join(f"{cell_mean[(n, ell)]:>+9.4f}" for ell in ells)
        )
    ranked = sorted(cell_mean.items(), key=lambda kv: kv[1])
    lines.append(
        "Weakest regimes (target these): "
        + ", ".join(f"N={n}/L={ell} {m:+.4f}" for (n, ell), m in ranked[:5])
    )
    winning = [((n, ell), m) for (n, ell), m in reversed(ranked) if m > 0.0][:5]
    if winning:
        lines.append(
            "Already winning: "
            + ", ".join(f"N={n}/L={ell} {m:+.4f}" for (n, ell), m in winning)
        )
    return "\n".join(lines)


def validate(solver):
    per_instance = _run_grid(solver)
    metrics = _metrics(per_instance)
    return metrics, _format_feedback(per_instance, metrics)
