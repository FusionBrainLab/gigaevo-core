"""Weighted bootstrap over a card's raw oriented gain deltas — the EV substrate.

The old EV bid was ``binarized_Beta_p_help x median_gain``: it threw away every
loss magnitude (a card that occasionally craters by −0.5 read identically to one
that dips by −0.001, as long as the sign counts matched) and priced the win at
the median, so a fat left tail was invisible to the auction. Bootstrap Thompson
prices the bid on the raw deltas instead: a card's expected-gain posterior is the
distribution of the mean of a weighted bootstrap resample of its own oriented
deltas plus one neutral pseudo-event. A genuinely cold card has no deltas, so it
uses the round's borrowed cold gain scale as its lone atom.

- The mean (not the median) is the statistic, so a fat left tail drags the bid
  down and confidently-negative cards fall below zero.
- Callers fold per-event staleness ``w_i = 2**(-s_i/H)`` into each delta's
  causal weight. As an event ages its delta fades toward neutral zero; a fresh
  event cannot revive older deltas. Positive cold scale is reserved for
  genuinely cold cards, so unrelated winners cannot make stale losing evidence
  bid positive.
- Each replicate draws the rounded Kish effective number of atoms from the
  exact sampling-weight vector, so fractional, skewed evidence has appropriately
  wider EV tails while zero-weight atoms add no false precision.
- One live draw is a Thompson sample of the card's EV; a batch of draws is the
  card's bootstrap-EV distribution, whose low quantile is the pessimistic EV the
  shortlist gate and the Tier-1 bench read (the successor to ``p_help_lo20``).

Pure numerics only — no card / config / task types leak in here.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib

import numpy as np


def _atoms_and_probs(
    deltas: Sequence[float],
    cold_scale: float,
    staleness_weight: float,
    delta_weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample support.

    Known cards: the card's ``k`` raw deltas (each sampling weight ``w``) plus one
    neutral pseudo-event valued at zero (unit weight), so stale evidence decays
    toward abstention. Cold cards (``k == 0``): the lone cold atom at
    ``cold_scale``, so their first injection stays explorable on the round's gain
    scale.
    """
    if not deltas:
        unit = np.asarray([1.0], dtype=float)
        return np.asarray([cold_scale], dtype=float), unit, unit
    atoms = np.asarray([*deltas, 0.0], dtype=float)
    weights = np.empty(atoms.shape[0], dtype=float)
    if delta_weights is not None and len(delta_weights) != len(deltas):
        raise ValueError(
            f"delta_weights length {len(delta_weights)} != deltas length {len(deltas)}"
        )
    local = (
        np.asarray(delta_weights, dtype=float)
        if delta_weights is not None
        else np.ones(len(deltas), dtype=float)
    )
    local = np.where(np.isfinite(local) & (local > 0.0), local, 0.0)
    weights[: len(deltas)] = max(staleness_weight, 0.0) * local
    weights[len(deltas)] = 1.0
    total = float(weights.sum())
    probs: np.ndarray
    if total <= 0.0:  # degenerate w and no cold weight cannot both happen, but guard
        probs = np.full_like(weights, 1.0 / atoms.shape[0], dtype=float)
    else:
        probs = weights / total
    return atoms, probs, weights


def _kish_resample_count(weights: np.ndarray) -> int:
    """Rounded Kish effective N for the exact atom weights being sampled.

    Scaling by the largest weight before evaluating ``(sum(w)**2) / sum(w**2)``
    avoids overflow without changing the ratio. Zero-weight atoms contribute to
    neither sum. The degenerate all-zero guard returns one draw rather than
    introducing a division-by-zero/NaN path.
    """
    scale = float(weights.max(initial=0.0))
    if not np.isfinite(scale) or scale <= 0.0:
        return 1
    normalized = weights / scale
    sum_weights = float(normalized.sum())
    sum_squares = float(np.square(normalized).sum())
    if sum_squares <= 0.0:
        return 1
    n_eff = sum_weights**2 / sum_squares
    return max(1, round(n_eff))


def _atom_ses(
    deltas: Sequence[float], ses: Sequence[float | None] | None, size: int
) -> np.ndarray | None:
    """Per-atom gain ses aligned with the resample support; the neutral
    pseudo-atom (and the cold atom) are exact (se=0). Unknown entries have no
    finite jitter scale. Returns ``None`` when no atom has a positive measured
    se, so the exact path consumes no extra rng — seed-exact replay of
    uncertainty-blind runs depends on this."""
    if ses is None or not deltas:
        return None
    if len(ses) != len(deltas):
        raise ValueError(f"ses length {len(ses)} != deltas length {len(deltas)}")
    vals = np.asarray([0.0 if se is None else float(se) for se in ses], dtype=float)
    arr = np.zeros(size, dtype=float)
    arr[: len(deltas)] = np.where(np.isfinite(vals) & (vals > 0.0), vals, 0.0)
    return arr if arr.any() else None


def bootstrap_ev_samples(
    deltas: Sequence[float],
    cold_scale: float,
    staleness_weight: float,
    n_samples: int,
    rng: np.random.Generator,
    *,
    delta_weights: Sequence[float] | None = None,
    ses: Sequence[float | None] | None = None,
) -> np.ndarray:
    """``n_samples`` bootstrap-resample means over the weighted delta support.

    ``n_samples`` remains the number of bootstrap replicates. Within each
    replicate, the atom draw count is ``max(1, round(n_eff))`` where
    ``n_eff = (sum(w)**2) / sum(w**2)`` over the exact sampling weights: fused
    ``delta_weights`` plus the unit neutral pseudo-event. A fixed-size
    multinomial bootstrap is used instead of a Poisson bootstrap because it
    preserves the historical all-unit shape and RNG stream exactly: all unit
    weights give ``n_eff == len(atoms)``, so the ``rng.choice`` call is byte-for-
    byte unchanged and no extra RNG is consumed. Zero weights contribute to
    neither Kish sum. Positive finite ``ses`` entries (aligned with ``deltas``)
    price measured evaluation noise by jittering each drawn atom by
    ``N(0, se)``; zero is exact and ``None`` has no measurable jitter scale.
    """
    atoms, probs, weights = _atoms_and_probs(
        deltas, cold_scale, staleness_weight, delta_weights
    )
    atom_count = atoms.shape[0]
    resample_count = _kish_resample_count(weights)
    idx = rng.choice(
        atom_count,
        size=(n_samples, resample_count),
        replace=True,
        p=probs,
    )
    picked = atoms[idx]
    atom_ses = _atom_ses(deltas, ses, atom_count)
    if atom_ses is not None:
        picked = picked + rng.normal(0.0, 1.0, size=picked.shape) * atom_ses[idx]
    return picked.mean(axis=1)


def bootstrap_ev_quantile(
    deltas: Sequence[float],
    cold_scale: float,
    staleness_weight: float,
    quantile: float,
    n_samples: int,
    rng: np.random.Generator,
    *,
    delta_weights: Sequence[float] | None = None,
) -> float:
    """The ``quantile`` of the card's bootstrap-EV distribution (e.g. the low
    quantile = pessimistic EV, the bootstrap successor to ``p_help_lo20``)."""
    samples = bootstrap_ev_samples(
        deltas,
        cold_scale,
        staleness_weight,
        n_samples,
        rng,
        delta_weights=delta_weights,
    )
    return float(np.quantile(samples, quantile))


def stable_rng(*keys: object) -> np.random.Generator:
    """A deterministic generator seeded from ``keys`` — for the per-card block-
    stats bootstrap, so a card's pessimistic EV is reproducible read-to-read and
    never consumes the live auction round's RNG."""
    digest = hashlib.sha256("|".join(str(k) for k in keys).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))
