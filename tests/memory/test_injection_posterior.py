"""Fix B bridge: per-program-card injection-efficacy posterior.

The Thompson auction's candidates are ``program-<uuid>`` cards. Their posterior
must be derived from how each card performed *when injected into a mutation
prompt*. Selection is stamped on the PARENT (``selected_ids``) and the child is
the outcome: the cards a child's mutation prompt actually contained are the
union of its parents' ``selected_ids``, so each such card receives the child's
parent-relative improvement as one event. A child's own ``selected_ids`` belong
to its future children and credit nothing here.
"""

from __future__ import annotations

import math

from pydantic import ValidationError
import pytest

from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.efficacy import beta_binomial_posterior
from gigaevo.memory.shared_memory.injection_posterior import (
    InjectionOutcome,
    compute_injection_posterior,
)
from gigaevo.memory.shared_memory.models import EvolutionStatistics

is_confidently_harmful = BetaBinomialReputation().is_confidently_harmful


def _stats(gains: list[float]) -> EvolutionStatistics:
    return EvolutionStatistics(ALL=beta_binomial_posterior(gains))


class TestIsConfidentlyHarmful:
    def test_three_all_harm_events_is_harmful(self) -> None:
        assert is_confidently_harmful(_stats([-0.01, -0.02, -0.03])) is True

    def test_two_all_harm_events_too_thin_to_exclude(self) -> None:
        assert is_confidently_harmful(_stats([-0.01, -0.02])) is False

    def test_mostly_helpful_card_not_harmful(self) -> None:
        assert is_confidently_harmful(_stats([0.01, 0.02, 0.03, -0.01])) is False

    def test_mixed_thin_evidence_not_harmful(self) -> None:
        assert is_confidently_harmful(_stats([0.01, -0.02, -0.03])) is False

    def test_missing_or_thin_stats_not_harmful(self) -> None:
        assert is_confidently_harmful(None) is False
        assert is_confidently_harmful(EvolutionStatistics()) is False
        assert (
            is_confidently_harmful(EvolutionStatistics.model_validate({"ALL": {}}))
            is False
        )
        assert (
            is_confidently_harmful(
                EvolutionStatistics.model_validate({"ALL": {"intro_events": 5}})
            )
            is False
        )

    def test_malformed_stats_rejected_at_validation_boundary(self) -> None:
        with pytest.raises(ValidationError):
            EvolutionStatistics.model_validate({"ALL": "garbage"})


def _prog(
    pid: str,
    *,
    fitness: float | None,
    parents: list[str] | None = None,
    selected: list[str] | None = None,
    injected: list[str] | None = None,
    invalid: bool = False,
) -> InjectionOutcome:
    return InjectionOutcome(
        id=pid,
        fitness=fitness,
        parents=parents or [],
        selected_ids=selected or [],
        injected_ids=injected,
        invalid=invalid,
    )


class TestBetaBinomialPosterior:
    def test_empty_gains_is_cold(self) -> None:
        post = beta_binomial_posterior([])
        assert post.posterior_a == 1.0
        assert post.posterior_b == 1.0
        assert post.intro_events == 0
        assert post.k_harm == 0
        assert math.isnan(post.p_help_lo20)
        assert post.efficacy_confident is False

    def test_single_help_event_not_yet_confident(self) -> None:
        post = beta_binomial_posterior([0.01])
        assert (post.posterior_a, post.posterior_b) == (2.0, 1.0)
        assert post.k_harm == 0
        assert post.p_help_lo20 == pytest.approx(0.4472, abs=1e-3)
        assert post.efficacy_confident is False

    def test_two_help_events_confident(self) -> None:
        post = beta_binomial_posterior([0.01, 0.02])
        assert (post.posterior_a, post.posterior_b) == (3.0, 1.0)
        assert post.p_help_lo20 == pytest.approx(0.5848, abs=1e-3)
        assert post.efficacy_confident is True

    def test_mixed_events_count_harm(self) -> None:
        post = beta_binomial_posterior([0.01, -0.02])
        assert (post.posterior_a, post.posterior_b) == (2.0, 2.0)
        assert post.k_harm == 1
        assert post.p_help_mean == pytest.approx(0.5)
        assert post.efficacy_confident is False

    def test_all_harm_is_suspect(self) -> None:
        post = beta_binomial_posterior([-0.01, -0.02, -0.03])
        assert (post.posterior_a, post.posterior_b) == (1.0, 4.0)
        assert post.k_harm == 3
        assert post.efficacy_confident is False

    def test_zero_gain_is_not_harm(self) -> None:
        # delta == 0 (no change) is not strictly harmful.
        post = beta_binomial_posterior([0.0, 0.0])
        assert post.k_harm == 0
        assert (post.posterior_a, post.posterior_b) == (3.0, 1.0)

    def test_nan_and_none_gains_filtered(self) -> None:
        post = beta_binomial_posterior([0.01, float("nan"), None, -0.02])  # type: ignore[list-item]
        assert post.intro_events == 2
        assert post.k_harm == 1

    def test_threshold_shifts_harm_count(self) -> None:
        # A negative threshold is a downside dead-band: only events below it count
        # as harm. Default threshold (0.0) preserves the strict < 0 behaviour.
        gains = [0.01, -0.001, -0.05]
        assert beta_binomial_posterior(gains).k_harm == 2
        assert beta_binomial_posterior(gains, threshold=-0.01).k_harm == 1


class TestComputeInjectionPosterior:
    def test_child_of_card_carrying_parent_credits_card(self) -> None:
        # Cards stamped on the parent are what the child's prompt contained.
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
            _prog("c1", fitness=0.85, parents=["root"], selected=[]),
        ]
        post = compute_injection_posterior(programs, higher_is_better=True)
        assert set(post) == {"program-A"}
        assert (post["program-A"].posterior_a, post["program-A"].posterior_b) == (
            2.0,
            1.0,
        )
        assert post["program-A"].k_harm == 0

    def test_childs_own_selected_ids_credit_nothing(self) -> None:
        # A child's own selected_ids feed its FUTURE children's prompts, not its
        # birth; crediting them measured selection bias one generation off.
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=[]),
            _prog("c1", fitness=0.85, parents=["root"], selected=["program-B"]),
        ]
        assert compute_injection_posterior(programs, higher_is_better=True) == {}

    def test_direction_flip_lower_is_better(self) -> None:
        # Lower fitness is better: child 0.10 beats parent 0.20 -> helpful.
        programs = [
            _prog("root", fitness=0.20, parents=[], selected=["program-A"]),
            _prog("c1", fitness=0.10, parents=["root"], selected=[]),
        ]
        post = compute_injection_posterior(programs, higher_is_better=False)
        assert post["program-A"].k_harm == 0
        assert post["program-A"].posterior_a == 2.0

    def test_uses_best_parent_among_multiple(self) -> None:
        # Child sits between two far-apart parents: vs the BEST parent (0.90) it is a
        # large genuine regression; vs the worst (0.60) it would look like a big gain.
        # A neutral sibling population (parented by the card-free 0.90 line) pins
        # baseline~0, so only the best-parent reference makes c1 harmful.
        siblings = [
            _prog(f"n{i}", fitness=0.90 + d, parents=["hi"], selected=[])
            for i, d in enumerate((0.0, 0.004, -0.004, 0.002, -0.002))
        ]
        programs = [
            _prog("lo", fitness=0.60, parents=[], selected=["program-A"]),
            _prog("hi", fitness=0.90, parents=[], selected=[]),
            *siblings,
            _prog("c1", fitness=0.70, parents=["lo", "hi"], selected=[]),
        ]
        post = compute_injection_posterior(programs, higher_is_better=True)
        assert post["program-A"].k_harm == 1

    def test_accumulates_events_across_children(self) -> None:
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
            _prog("c1", fitness=0.85, parents=["root"], selected=[]),
            _prog("c2", fitness=0.86, parents=["root"], selected=[]),
        ]
        post = compute_injection_posterior(programs, higher_is_better=True)
        assert post["program-A"].intro_events == 2
        assert post["program-A"].efficacy_confident is True

    def test_child_without_resolvable_parent_skipped(self) -> None:
        # Parent id not present in the program set -> no baseline -> skipped.
        programs = [
            _prog("c1", fitness=0.85, parents=["ghost"], selected=[]),
        ]
        assert compute_injection_posterior(programs, higher_is_better=True) == {}

    def test_parent_with_no_selected_ids_contributes_nothing(self) -> None:
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=[]),
            _prog("c1", fitness=0.85, parents=["root"], selected=[]),
        ]
        assert compute_injection_posterior(programs, higher_is_better=True) == {}

    def test_child_with_missing_fitness_skipped(self) -> None:
        # Missing fitness (never evaluated / metric absent) is no evidence.
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
            _prog("c1", fitness=None, parents=["root"], selected=[]),
        ]
        assert compute_injection_posterior(programs, higher_is_better=True) == {}

    def test_invalid_child_counts_as_harm_event(self) -> None:
        # A child evaluated and judged invalid is the worst possible outcome of
        # an injection — it must raise k_harm even without a gain magnitude.
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
            _prog("c1", fitness=None, parents=["root"], invalid=True),
        ]
        post = compute_injection_posterior(programs, higher_is_better=True)
        assert post["program-A"].intro_events == 1
        assert post["program-A"].k_harm == 1
        assert (post["program-A"].posterior_a, post["program-A"].posterior_b) == (
            1.0,
            2.0,
        )

    def test_invalid_child_without_valid_parent_skipped(self) -> None:
        programs = [
            _prog("root", fitness=None, parents=[], selected=["program-A"]),
            _prog("c1", fitness=None, parents=["root"], invalid=True),
        ]
        assert compute_injection_posterior(programs, higher_is_better=True) == {}

    def test_invalid_child_never_enters_the_baseline_cohort(self) -> None:
        # The invalid child contributes a harm event but no gain — siblings'
        # baselines must be computed as if it produced no number at all.
        with_invalid = [
            _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
            _prog("c1", fitness=0.85, parents=["root"]),
            _prog("c2", fitness=None, parents=["root"], invalid=True),
        ]
        without_invalid = [
            _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
            _prog("c1", fitness=0.85, parents=["root"]),
        ]
        a = compute_injection_posterior(with_invalid, higher_is_better=True)
        b = compute_injection_posterior(without_invalid, higher_is_better=True)
        assert a["program-A"].k_harm == b["program-A"].k_harm + 1
        assert a["program-A"].intro_events == b["program-A"].intro_events + 1

    def test_never_injected_card_absent(self) -> None:
        # program-Z is in nobody's parent selected set -> absent (auction: COLD).
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
            _prog("c1", fitness=0.85, parents=["root"], selected=[]),
        ]
        post = compute_injection_posterior(programs, higher_is_better=True)
        assert "program-Z" not in post


class TestRedesignedHarmDefinition:
    """Harm = below the parent-fitness-local counterfactual by more than the
    data-derived noise band, not the raw < 0 regression. Pins the redesign.
    Card exposure rides on a dedicated parent (rootA) at the same fitness as the
    card-free root so both lines share one baseline."""

    def test_within_noise_regressions_are_not_harm(self) -> None:
        # A spread-out population sets a noise band well above the card's tiny
        # sub-noise regressions, so the card accrues no harm and turns confident.
        spread = [
            _prog(f"s{i}", fitness=0.80 + d, parents=["root"], selected=[])
            for i, d in enumerate((0.10, -0.10, 0.08, -0.08, 0.05, -0.05))
        ]
        card = [
            _prog(f"c{i}", fitness=0.80 + d, parents=["rootA"], selected=[])
            for i, d in enumerate((-0.005, -0.007, -0.004))
        ]
        post = compute_injection_posterior(
            [
                _prog("root", fitness=0.80, parents=[], selected=[]),
                _prog("rootA", fitness=0.80, parents=[], selected=["program-A"]),
                *spread,
                *card,
            ],
            higher_is_better=True,
        )
        assert post["program-A"].k_harm == 0
        assert post["program-A"].efficacy_confident is True

    def test_plateau_card_matching_baseline_is_neutral(self) -> None:
        # Every mutation regresses at the plateau; a card whose children regress by
        # the typical local amount is neutral (not SUSPECT), unlike the raw metric.
        plateau = [
            _prog(f"q{i}", fitness=0.855 + d, parents=["root"], selected=[])
            for i, d in enumerate((-0.020, -0.018, -0.022, -0.019, -0.021, -0.020))
        ]
        card = [
            _prog(f"c{i}", fitness=0.855 + d, parents=["rootA"], selected=[])
            for i, d in enumerate((-0.020, -0.019, -0.021))
        ]
        post = compute_injection_posterior(
            [
                _prog("root", fitness=0.855, parents=[], selected=[]),
                _prog("rootA", fitness=0.855, parents=[], selected=["program-A"]),
                *plateau,
                *card,
            ],
            higher_is_better=True,
        )
        assert post["program-A"].k_harm == 0
        assert post["program-A"].efficacy_confident is True

    def test_genuine_consistent_regression_is_harm(self) -> None:
        # Children fall far below the local counterfactual: real harm survives.
        neutral = [
            _prog(f"m{i}", fitness=0.80 + d, parents=["root"], selected=[])
            for i, d in enumerate((0.005, -0.005, 0.003, -0.003, 0.0, 0.002))
        ]
        card = [
            _prog(f"c{i}", fitness=f, parents=["rootA"], selected=[])
            for i, f in enumerate((0.68, 0.67, 0.69))
        ]
        post = compute_injection_posterior(
            [
                _prog("root", fitness=0.80, parents=[], selected=[]),
                _prog("rootA", fitness=0.80, parents=[], selected=["program-A"]),
                *neutral,
                *card,
            ],
            higher_is_better=True,
        )
        assert post["program-A"].k_harm == 3
        assert post["program-A"].efficacy_confident is False

    def test_minimisation_genuine_harm_still_suspect(self) -> None:
        # Lower fitness is better; card children land far ABOVE (worse) the parent.
        neutral = [
            _prog(f"m{i}", fitness=0.50 + d, parents=["root"], selected=[])
            for i, d in enumerate((0.005, -0.005, 0.003, -0.003, 0.0, 0.002))
        ]
        card = [
            _prog(f"c{i}", fitness=f, parents=["rootA"], selected=[])
            for i, f in enumerate((0.62, 0.63, 0.61))
        ]
        post = compute_injection_posterior(
            [
                _prog("root", fitness=0.50, parents=[], selected=[]),
                _prog("rootA", fitness=0.50, parents=[], selected=["program-A"]),
                *neutral,
                *card,
            ],
            higher_is_better=False,
        )
        assert post["program-A"].k_harm == 3

    def test_discrete_unit_steps_register_without_a_guard(self) -> None:
        # Quantised fitness, mostly-flat mutations -> noise band collapses to 0, so a
        # clean +1 step is helpful and a -1 step is genuine harm (nothing swallowed).
        flat = [
            _prog(f"f{i}", fitness=0.50, parents=["root"], selected=[])
            for i in range(6)
        ]
        up = [
            _prog(f"u{i}", fitness=0.55, parents=["rootUP"], selected=[])
            for i in range(2)
        ]
        down = [
            _prog(f"d{i}", fitness=0.45, parents=["rootDOWN"], selected=[])
            for i in range(2)
        ]
        post = compute_injection_posterior(
            [
                _prog("root", fitness=0.50, parents=[], selected=[]),
                _prog("rootUP", fitness=0.50, parents=[], selected=["program-UP"]),
                _prog("rootDOWN", fitness=0.50, parents=[], selected=["program-DOWN"]),
                *flat,
                *up,
                *down,
            ],
            higher_is_better=True,
        )
        assert post["program-UP"].k_harm == 0
        assert post["program-DOWN"].k_harm == 2

    def test_deterministic_under_input_reordering(self) -> None:
        # median + MAD are deterministic; the live signal must equal the offline
        # reconstruction, which sees programs in a different order.
        spread = [
            _prog(f"s{i}", fitness=0.80 + d, parents=["root"], selected=[])
            for i, d in enumerate((0.10, -0.10, 0.08, -0.08, 0.05, -0.05))
        ]
        card = [
            _prog(f"c{i}", fitness=0.80 + d, parents=["rootA"], selected=[])
            for i, d in enumerate((-0.005, -0.12, -0.004))
        ]
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=[]),
            _prog("rootA", fitness=0.80, parents=[], selected=["program-A"]),
            *spread,
            *card,
        ]
        forward = compute_injection_posterior(programs, higher_is_better=True)
        backward = compute_injection_posterior(
            list(reversed(programs)), higher_is_better=True
        )
        assert forward == backward


class TestBirthStampAttribution:
    """Attribution reads the child's frozen birth slate (``injected_ids``),
    falling back to the parents' current ``selected_ids`` only for legacy
    programs born before the stamp existed."""

    def test_frozen_birth_slate_overrides_parent_current_slate(self) -> None:
        # Parent re-ran the stochastic auction after the child was born; the
        # child's prompt contained program-OLD, not the parent's current pick.
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=["program-NEW"]),
            _prog("c1", fitness=0.85, parents=["root"], injected=["program-OLD"]),
        ]
        post = compute_injection_posterior(programs, higher_is_better=True)
        assert set(post) == {"program-OLD"}

    def test_empty_birth_slate_means_no_exposure(self) -> None:
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
            _prog("c1", fitness=0.85, parents=["root"], injected=[]),
        ]
        assert compute_injection_posterior(programs, higher_is_better=True) == {}

    def test_legacy_child_without_stamp_falls_back_to_parent_slates(self) -> None:
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
            _prog("c1", fitness=0.85, parents=["root"], injected=None),
        ]
        post = compute_injection_posterior(programs, higher_is_better=True)
        assert set(post) == {"program-A"}


class TestNoiseFloor:
    def test_zero_mad_plateau_jitter_below_floor_is_not_harm(self) -> None:
        # Over half the cohort ties exactly at the median -> MAD collapses to
        # zero; sub-floor jitter (CV fold noise) must not register as harm.
        siblings = [_prog(f"n{i}", fitness=0.80, parents=["root"]) for i in range(5)]
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=[]),
            _prog("rootA", fitness=0.80, parents=[], selected=["program-A"]),
            *siblings,
            _prog("c1", fitness=0.80 - 1e-6, parents=["rootA"]),
        ]
        post = compute_injection_posterior(programs, higher_is_better=True)
        assert post["program-A"].k_harm == 0


class TestLeaveOneOutBaseline:
    def test_lone_regressing_child_is_harm_not_help(self) -> None:
        # With self-inclusion the child IS its own baseline (adjusted gain 0),
        # so a hard regression scored as help. Leave-one-out exposes it.
        programs = [
            _prog("root", fitness=0.80, parents=[], selected=["program-A"]),
            _prog("c1", fitness=0.70, parents=["root"]),
        ]
        post = compute_injection_posterior(programs, higher_is_better=True)
        assert post["program-A"].k_harm == 1
