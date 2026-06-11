"""Per-program-card injection-efficacy posterior (Fix B bridge).

The Thompson auction (``run_card_auction``) draws each candidate's downside
Beta-Binomial posterior, but its candidates are ``program-<uuid>`` cards. Their
posterior must therefore be keyed in that id-space and derived from how a card
performed *when injected into a mutation prompt*. Card selection is stamped on
the PARENT (``selected_ids``) and the child is the outcome: the cards a child's
prompt actually contained are the union of its parents' ``selected_ids``, so
each such card receives the child's parent-relative improvement as one event.
(A child's own ``selected_ids`` feed its future children's prompts and credit
nothing at its own birth — crediting them would measure selection bias one
generation off.)

Harm is judged *relative to the parent-fitness-local counterfactual* and only
beyond a *data-derived noise band*: a child counts as harmful for a card iff its
improvement falls below the typical improvement of equally-fit parents' mutations
by more than the population's robust noise scale. This avoids labelling every
sub-ceiling or sub-noise regression as harm (which conflates "search near its
ceiling" with "card is bad").

This module is the single numeric implementation; ``BetaBinomialReputation``
(gigaevo/memory/core/reputation.py) is the injectable façade that binds its
configured thresholds to these primitives.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
import statistics
from typing import Any

from loguru import logger
from scipy.stats import beta

_BASELINE_NEIGHBORS = 15
_NOISE_BAND_K = 1.0
_CONFIDENT_QUANTILE = 0.20
_CONFIDENT_THRESHOLD = 0.5
_MAD_TO_SIGMA = 1.4826


def beta_binomial_posterior(
    gains: Sequence[float],
    *,
    threshold: float = 0.0,
    confident_quantile: float = _CONFIDENT_QUANTILE,
    confident_threshold: float = _CONFIDENT_THRESHOLD,
) -> dict[str, Any]:
    """Downside Beta-Binomial posterior on P(not harmful) from per-event gains.

    ``a = 1 + (n - k_harm)``, ``b = 1 + k_harm`` with ``k_harm`` the count of
    events whose gain is below ``threshold`` (default 0); ``efficacy_confident``
    iff the ``confident_quantile`` of Beta(a, b) exceeds ``confident_threshold``.
    The ``p_help_lo20`` key name is part of the banks.json contract regardless
    of the configured quantile.
    """
    finite = [float(g) for g in gains if g is not None and math.isfinite(float(g))]
    n = len(finite)
    k_harm = sum(1 for g in finite if g < threshold)
    a = 1.0 + (n - k_harm)
    b = 1.0 + k_harm
    lo = float(beta.ppf(confident_quantile, a, b)) if n else float("nan")
    return {
        "posterior_a": a,
        "posterior_b": b,
        "intro_events": n,
        "k_harm": k_harm,
        "p_help_mean": a / (a + b),
        "p_help_lo20": lo,
        "efficacy_confident": bool(n and lo > confident_threshold),
    }


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def parent_local_baseline(
    population: Sequence[tuple[float, float]],
    *,
    neighbors: int = _BASELINE_NEIGHBORS,
) -> Callable[[float], float]:
    """Counterfactual improvement for a parent of fitness ``f``: the median gain of
    the ``neighbors`` scorable children with the nearest best-parent fitness.
    With few children every point is used (a global median)."""
    refs = [r for r, _ in population]
    gains = [g for _, g in population]
    n = len(gains)
    k = min(neighbors, n)
    if not n:
        return lambda f: 0.0

    def baseline(f: float) -> float:
        nearest = sorted(range(n), key=lambda i: abs(refs[i] - f))[:k]
        return _median([gains[i] for i in nearest])

    return baseline


def noise_band(centered: Sequence[float]) -> float:
    """Robust noise scale of the centred improvements (MAD → σ). Collapses to 0 for
    a degenerate/flat population so genuine discrete steps still register."""
    if not centered:
        return 0.0
    med = _median(centered)
    mad = _median([abs(x - med) for x in centered])
    return _MAD_TO_SIGMA * mad


def compute_injection_posterior(
    programs: Sequence[Mapping[str, Any]],
    *,
    higher_is_better: bool = True,
    baseline_neighbors: int = _BASELINE_NEIGHBORS,
    noise_band_k: float = _NOISE_BAND_K,
    confident_quantile: float = _CONFIDENT_QUANTILE,
    confident_threshold: float = _CONFIDENT_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Map each injected card id to its injection posterior.

    For each child program, every card in the union of its resolvable parents'
    ``selected_ids`` — the cards actually present in the mutation prompt that
    produced the child — receives one intro event. The event's gain is the
    parent-relative improvement *minus* the parent-fitness-local counterfactual;
    it counts as harm only if it falls below the population's noise band. Cards
    never injected into a child with a valid parent baseline are absent from the
    result, which the auction treats as COLD Beta(1, 1).
    """
    by_id = {
        str(p["id"]): p for p in programs if isinstance(p, Mapping) and p.get("id")
    }
    population: list[tuple[float, float]] = []
    events: dict[str, list[tuple[float, float]]] = {}
    for p in programs:
        if not isinstance(p, Mapping):
            continue
        fitness = p.get("fitness")
        if fitness is None:
            continue
        parent_union: set[str] = set()
        parent_fits: list[float] = []
        for par_id in p.get("parents") or []:
            parent = by_id.get(str(par_id))
            if parent is None:
                continue
            parent_union |= {str(c) for c in (parent.get("selected_ids") or []) if c}
            par_fit = parent.get("fitness")
            if par_fit is not None:
                parent_fits.append(float(par_fit))
        if not parent_fits:
            continue
        ref = max(parent_fits) if higher_is_better else min(parent_fits)
        gain = float(fitness) - ref if higher_is_better else ref - float(fitness)
        population.append((ref, gain))
        for card_id in parent_union:
            events.setdefault(card_id, []).append((gain, ref))

    if not events:
        return {}

    baseline = parent_local_baseline(population, neighbors=baseline_neighbors)
    epsilon = noise_band_k * noise_band([g - baseline(ref) for ref, g in population])
    posteriors = {
        card_id: beta_binomial_posterior(
            [g - baseline(ref) for g, ref in evs],
            threshold=-epsilon,
            confident_quantile=confident_quantile,
            confident_threshold=confident_threshold,
        )
        for card_id, evs in events.items()
    }
    logger.debug(
        "[Memory][InjectionPosterior] {} card(s) credited from {} scorable child(ren); "
        "noise band epsilon={:.4g}, confident={}",
        len(posteriors),
        len(population),
        epsilon,
        sum(1 for p in posteriors.values() if p["efficacy_confident"]),
    )
    return posteriors
