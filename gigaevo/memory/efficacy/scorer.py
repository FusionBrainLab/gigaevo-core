"""Owner of the gain -> downside-posterior math.

``block_from_events`` turns a card's gain events into the global, unadjusted
reputation block (median magnitude, MAD harm band) that the auction and renderer
read. It is built on the shared ``beta_binomial_posterior`` / ``robust_noise_band``
core, which the BD in-cell partition reuses over its own event subset.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import statistics

from scipy.stats import beta

from gigaevo.memory.context import ContextualGain
from gigaevo.memory.shared_memory.models import CardStatsBlock

_MAD_TO_SIGMA = 1.4826


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def robust_noise_band(values: Sequence[float]) -> float:
    """Robust noise scale of the values (MAD -> sigma), centred on their median.
    Collapses to 0 for a degenerate/flat set so genuine discrete steps still
    register. The single source of the harm-predicate noise band for both the
    global counterfactual path and the BD-cell partition."""
    if not values:
        return 0.0
    med = _median(values)
    return _MAD_TO_SIGMA * _median([abs(x - med) for x in values])


def beta_binomial_posterior(
    gains: Sequence[float],
    *,
    threshold: float = 0.0,
    invalid_events: int = 0,
    confident_quantile: float = 0.20,
    confident_threshold: float = 0.5,
) -> CardStatsBlock:
    """Downside Beta-Binomial posterior on P(not harmful) from per-event gains.

    ``a = 1 + (n - k_harm)``, ``b = 1 + k_harm`` with ``k_harm`` the count of
    events whose gain is below ``threshold`` (default 0); ``efficacy_confident``
    iff the ``confident_quantile`` of Beta(a, b) exceeds ``confident_threshold``.
    The ``p_help_lo20`` field name is part of the serialized-card stats contract
    regardless of the configured quantile. ``invalid_events`` are evaluated-and-judged-
    invalid children: each is one forced harm event with no gain magnitude.
    """
    finite = [float(g) for g in gains if g is not None and math.isfinite(float(g))]
    n = len(finite) + invalid_events
    k_harm = sum(1 for g in finite if g < threshold) + invalid_events
    a = 1.0 + (n - k_harm)
    b = 1.0 + k_harm
    lo = float(beta.ppf(confident_quantile, a, b)) if n else float("nan")
    return CardStatsBlock(
        posterior_a=a,
        posterior_b=b,
        intro_events=n,
        k_harm=k_harm,
        p_help_mean=a / (a + b),
        p_help_lo20=lo,
        efficacy_confident=bool(n and lo > confident_threshold),
    )


def block_from_events(
    events: Sequence[ContextualGain],
    *,
    noise_band_k: float = 1.0,
    confident_quantile: float = 0.20,
    confident_threshold: float = 0.5,
) -> CardStatsBlock | None:
    """Global, unadjusted card block from its gain events: median magnitude plus
    the downside posterior, harm being a gain below the robust noise band
    ``-noise_band_k * MAD`` of the finite valid gains. Invalid events are forced
    harm with no magnitude. Returns ``None`` for a card with no events (no
    evidence, no block). The single owner of the events -> card block math,
    shared by the global reputation and the BD in-cell partition.
    """
    if not events:
        return None
    valid = [e for e in events if not e.invalid]
    invalid_events = len(events) - len(valid)
    valid_gains = [float(e.gain) for e in valid]
    finite_gains = [g for g in valid_gains if math.isfinite(g)]
    epsilon = noise_band_k * robust_noise_band(finite_gains)
    block = beta_binomial_posterior(
        valid_gains,
        threshold=-epsilon,
        invalid_events=invalid_events,
        confident_quantile=confident_quantile,
        confident_threshold=confident_threshold,
    )
    magnitude = _median(finite_gains) if finite_gains else 0.0
    return block.model_copy(update={"IntroGain_best_median": magnitude})
