from __future__ import annotations

from collections.abc import Callable

import pytest

from gigaevo.memory.cards import Card, CardUseTrial
from gigaevo.memory_v2.models import (
    CardSnapshot,
    CausalObservation,
    OutcomeMeasurement,
)
from gigaevo.memory_v2.posterior import TerminalUtilityPosteriorConfig
from gigaevo.memory_v2.transfer import (
    CrossTaskUsefulnessConfig,
    CrossTaskUsefulnessModel,
)
from gigaevo.memory_v2.writer import CausalV2ContentOnlyUpdater


def _trials(
    card_id: str,
    task_key: str,
    *,
    treatment_success: bool,
    run_id: str | None = None,
) -> tuple[CardUseTrial, ...]:
    run = run_id or f"run-{task_key}"
    return tuple(
        CardUseTrial(
            decision_id=f"{card_id}-{task_key}-{arm}-{repeat}",
            run_id=run,
            task_key=task_key,
            treatment=arm == "treated",
            success=treatment_success if arm == "treated" else not treatment_success,
        )
        for arm in ("control", "treated")
        for repeat in range(4)
    )


def _arm_trials(
    card_id: str,
    task_key: str,
    run_id: str,
    *,
    treatment: bool,
    successes: int,
    total: int,
) -> tuple[CardUseTrial, ...]:
    return tuple(
        CardUseTrial(
            decision_id=f"{card_id}-{run_id}-{treatment}-{index}",
            run_id=run_id,
            task_key=task_key,
            treatment=treatment,
            success=index < successes,
        )
        for index in range(total)
    )


def test_hierarchical_usefulness_transfers_binary_direction_across_tasks() -> None:
    good = Card(
        id="good",
        description="good idea",
        use_trials=(
            *_trials("good", "small-loss", treatment_success=True),
            *_trials("good", "huge-loss", treatment_success=True),
        ),
    )
    bad = Card(
        id="bad",
        description="bad idea",
        use_trials=(
            *_trials("bad", "small-loss", treatment_success=False),
            *_trials("bad", "huge-loss", treatment_success=False),
        ),
    )
    model = CrossTaskUsefulnessModel(
        CrossTaskUsefulnessConfig(minimum_trials_per_arm=2),
        TerminalUtilityPosteriorConfig(),
    )

    fitted = model.fit(
        (good, bad), target_task_key="new-task", current_run_id="active-run"
    )

    assert fitted.observations == 32
    assert fitted.helpful_probability["good"] > 0.65
    assert fitted.helpful_probability["bad"] < 0.35


def test_run_strata_prevent_simpson_bias_from_different_offer_mixes() -> None:
    card = Card(
        id="card",
        description="neutral idea",
        use_trials=(
            # Both arms succeed 75% in the high-baseline run, but treatment is
            # offered more often there.
            *_arm_trials("card", "task", "high", treatment=False, successes=3, total=4),
            *_arm_trials("card", "task", "high", treatment=True, successes=9, total=12),
            # Both arms succeed 25% in the low-baseline run, where control is
            # more common. Pooling runs would manufacture a positive effect.
            *_arm_trials("card", "task", "low", treatment=False, successes=3, total=12),
            *_arm_trials("card", "task", "low", treatment=True, successes=1, total=4),
        ),
    )
    model = CrossTaskUsefulnessModel(
        CrossTaskUsefulnessConfig(minimum_trials_per_arm=2),
        TerminalUtilityPosteriorConfig(),
    )

    fitted = model.fit((card,), target_task_key="task", current_run_id="active-run")

    assert fitted.observations == 32
    assert fitted.helpful_probability["card"] == pytest.approx(0.5, abs=0.08)


def test_current_run_trials_are_left_to_the_local_posterior() -> None:
    card = Card(
        id="card",
        description="idea",
        use_trials=_trials(
            "card", "target", treatment_success=True, run_id="active-run"
        ),
    )
    model = CrossTaskUsefulnessModel(
        CrossTaskUsefulnessConfig(), TerminalUtilityPosteriorConfig()
    )

    fitted = model.fit((card,), target_task_key="target", current_run_id="active-run")

    assert fitted.observations == 0
    assert fitted.helpful_probability == {"card": 0.5}


def test_cross_task_prior_moves_only_reward_card_intercept(
    posterior_model,
) -> None:
    revision = CardSnapshot.from_card(Card(id="card", description="idea"))
    neutral = posterior_model.fit((), (revision,))
    shifted = posterior_model.fit(
        (), (revision,), card_intercept_prior_mean={"card": 0.1}
    )
    coefficient = shifted.space.card_intercept_index("card")

    assert neutral.reward.mean[coefficient] == pytest.approx(0.0)
    assert shifted.reward.mean[coefficient] == pytest.approx(0.1)
    assert shifted.reward.covariance == pytest.approx(neutral.reward.covariance)
    assert shifted.safety.mean == pytest.approx(neutral.safety.mean)
    assert shifted.lineage_reward.mean == pytest.approx(neutral.lineage_reward.mean)


class _Store:
    def __init__(self, card: Card) -> None:
        self.card = card

    def snapshot(self) -> tuple[Card, ...]:
        return (self.card,)

    def update(
        self, card_id: str, transform: Callable[[Card], Card | None]
    ) -> Card | None:
        if card_id != self.card.id:
            return None
        replacement = transform(self.card)
        if replacement is not None:
            self.card = replacement
        return replacement


def test_writer_folds_scale_free_randomized_trials_onto_card(
    evolution_context,
) -> None:
    card = Card(id="card", description="idea")
    revision = CardSnapshot.from_card(card)
    treated = CausalObservation(
        decision_id="treated",
        event_ordinal=0,
        card=revision,
        context=evolution_context,
        treatment=True,
        card_used=False,
        offer_propensity=0.5,
        proposal_propensity=1.0,
        joint_action_propensity=0.5,
        status="outcome",
        measurement=OutcomeMeasurement(value=0.01, se=None, kind="scalar"),
        reward_q_hat_control=0.0,
        reward_q_hat_treated=0.0,
        risk_q_hat_control=0.05,
        risk_q_hat_treated=0.05,
    )
    control = treated.model_copy(
        update={
            "decision_id": "control",
            "event_ordinal": 1,
            "treatment": False,
            "status": "invalid",
            "measurement": None,
        }
    )
    store = _Store(card)

    CausalV2ContentOnlyUpdater._record_use_trials(
        (treated, control),
        store=store,  # type: ignore[arg-type]
    )
    CausalV2ContentOnlyUpdater._record_use_trials(
        (treated, control),
        store=store,  # type: ignore[arg-type]
    )

    assert len(store.card.use_trials) == 2
    by_id = {trial.decision_id: trial for trial in store.card.use_trials}
    assert by_id["treated"].treatment is True
    assert by_id["treated"].success is True
    assert by_id["control"].treatment is False
    assert by_id["control"].success is False
    assert {trial.task_key for trial in store.card.use_trials} == {"task"}
