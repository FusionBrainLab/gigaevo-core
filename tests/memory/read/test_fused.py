"""FusedRankingShortlister: semantic-rank + reputation + novelty reordering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from omegaconf import OmegaConf
import pytest

from gigaevo.memory.cards import CardStatsBlock, ContextualGain, DecisionContext
from gigaevo.memory.read.fused import (
    BootstrapFusedRankingShortlister,
    FusedRankingShortlister,
)
from gigaevo.memory.read.reputation import BetaBinomialReputation, BootstrapReputation
from gigaevo.memory.storage.base import ResearchResult

_REPO_ROOT = Path(__file__).resolve().parents[3]


class StubShortlister:
    def __init__(self, result: ResearchResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def shortlist(self, **kwargs) -> ResearchResult:
        self.calls.append(kwargs)
        return self.result


class RecordingReputation(BetaBinomialReputation):
    contexts: ClassVar[list] = []

    def card_stats(self, card, context=None):
        type(self).contexts.append(context)
        return super().card_stats(card, context)


class BootstrapEVStubReputation:
    policy_min_effective_events = 3.0

    def __init__(self, ev_by_id: dict[str, float]) -> None:
        self._ev_by_id = ev_by_id

    def card_stats(self, card, context=None):
        del context
        ev = self._ev_by_id.get(card.id)
        if ev is None:
            return None
        return CardStatsBlock(
            posterior_a=2.0,
            posterior_b=1.0,
            intro_events=1,
            p_help_lo20=0.8,
            IntroGain_bootstrap_ev_mean=ev,
            IntroGain_bootstrap_ev_lo20=ev,
            IntroGain_bootstrap_ev_hi80=ev,
        )

    def magnitude_of(self, block):
        if block is None:
            return None
        return block.IntroGain_bootstrap_ev_mean

    def event_deltas(self, card, context=None):
        del context
        return tuple(
            float(event.gain)
            for event in card.gain_events
            if not event.invalid and not event.founding and not event.unused
        )

    def event_weights(self, card, context=None):
        return tuple(1.0 for _ in self.event_deltas(card, context))

    def staleness_weights(self, card, context=None):
        return tuple(1.0 for _ in self.event_deltas(card, context))


def _parents() -> list:
    return [SimpleNamespace(id="parent-1", metrics={"score": 1.5})]


def _shortlist_kwargs() -> dict:
    return {
        "parents": _parents(),
        "mutation_mode": "rewrite",
        "task_description": "task",
        "metrics_description": "metrics",
    }


def _stamped_event(gain: float, *, hours_ago: float) -> ContextualGain:
    return ContextualGain(
        context=DecisionContext(
            timestamp=datetime.now(UTC) - timedelta(hours=hours_ago)
        ),
        gain=gain,
    )


async def test_neutral_weights_return_inner_result_untouched(make_card, make_event):
    result = ResearchResult(
        cards=(make_card(gain_events=(make_event(-1.0),)), make_card()),
        summary="why",
        iterations=2,
    )
    inner = StubShortlister(result)
    fused = FusedRankingShortlister(
        inner=inner, reputation=BetaBinomialReputation(), w_rep=0.0, w_nov=0.0
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert out is result
    assert inner.calls[0]["mutation_mode"] == "rewrite"


async def test_w_rep_reorders_by_p_help_lo20(make_card, make_event):
    bad = make_card(gain_events=tuple(make_event(-1.0) for _ in range(4)))
    good = make_card(gain_events=tuple(make_event(1.0) for _ in range(4)))
    inner = StubShortlister(
        ResearchResult(cards=(bad, good), summary="why", iterations=1)
    )
    fused = FusedRankingShortlister(
        inner=inner,
        reputation=BetaBinomialReputation(),
        w_sem=0.0,
        w_rep=1.0,
        w_nov=0.0,
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert [card.id for card in out.cards] == [good.id, bad.id]
    assert out.summary == "why"
    assert out.iterations == 1


async def test_cards_without_evidence_score_neutral(make_card, make_event):
    bad = make_card(gain_events=tuple(make_event(-1.0) for _ in range(4)))
    cold = make_card()
    good = make_card(gain_events=tuple(make_event(1.0) for _ in range(4)))
    inner = StubShortlister(ResearchResult(cards=(bad, cold, good)))
    fused = FusedRankingShortlister(
        inner=inner,
        reputation=BetaBinomialReputation(),
        w_sem=0.0,
        w_rep=1.0,
        w_nov=0.0,
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert [card.id for card in out.cards] == [good.id, cold.id, bad.id]


async def test_w_nov_demotes_recently_used_card(make_card):
    hot = make_card(
        gain_events=tuple(_stamped_event(1.0, hours_ago=1.0) for _ in range(3))
    )
    rested = make_card(gain_events=(_stamped_event(1.0, hours_ago=100.0),))
    inner = StubShortlister(ResearchResult(cards=(hot, rested)))
    fused = FusedRankingShortlister(
        inner=inner,
        reputation=BetaBinomialReputation(),
        w_sem=0.0,
        w_rep=0.0,
        w_nov=1.0,
        novelty_window_hours=24.0,
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert [card.id for card in out.cards] == [rested.id, hot.id]


async def test_ties_keep_inner_order(make_card):
    first = make_card()
    second = make_card()
    inner = StubShortlister(ResearchResult(cards=(first, second)))
    fused = FusedRankingShortlister(
        inner=inner,
        reputation=BetaBinomialReputation(),
        w_sem=0.0,
        w_rep=1.0,
        w_nov=0.0,
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert [card.id for card in out.cards] == [first.id, second.id]


async def test_context_mirrors_primary_parent_metrics(make_card, make_event):
    RecordingReputation.contexts.clear()
    inner = StubShortlister(
        ResearchResult(cards=(make_card(gain_events=(make_event(1.0),)),))
    )
    fused = FusedRankingShortlister(
        inner=inner, reputation=RecordingReputation(), w_rep=1.0
    )
    await fused.shortlist(**_shortlist_kwargs())
    assert RecordingReputation.contexts == [
        DecisionContext(parent_metrics={"score": 1.5}, parent_id="parent-1")
    ]


async def test_score_floor_drops_harm_marked_keeps_cold_and_clean(
    make_card, make_event
):
    harm = make_card(gain_events=tuple(make_event(-1.0) for _ in range(4)))
    cold = make_card()
    clean_single = make_card(gain_events=(make_event(1.0),))
    inner = StubShortlister(ResearchResult(cards=(harm, cold, clean_single)))
    fused = FusedRankingShortlister(
        inner=inner,
        reputation=BetaBinomialReputation(),
        w_sem=0.0,
        w_rep=1.0,
        w_nov=0.0,
        score_floor=0.44,
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert [card.id for card in out.cards] == [cold.id, clean_single.id]


async def test_score_floor_empty_when_no_card_clears(make_card, make_event):
    inner = StubShortlister(
        ResearchResult(
            cards=(make_card(gain_events=(make_event(-1.0),)),),
            summary="why",
            iterations=2,
        )
    )
    fused = FusedRankingShortlister(
        inner=inner,
        reputation=BetaBinomialReputation(),
        w_sem=0.0,
        w_rep=1.0,
        w_nov=0.0,
        score_floor=0.44,
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert out.cards == ()
    assert out.summary == "why"
    assert out.iterations == 2


async def test_score_floor_applies_even_at_neutral_rep_nov_weights(make_card):
    first, second, third = make_card(), make_card(), make_card()
    inner = StubShortlister(ResearchResult(cards=(first, second, third)))
    fused = FusedRankingShortlister(
        inner=inner,
        reputation=BetaBinomialReputation(),
        w_sem=1.0,
        w_rep=0.0,
        w_nov=0.0,
        score_floor=0.75,
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert [card.id for card in out.cards] == [first.id]


class StubStore:
    def __init__(self, cards: tuple) -> None:
        self._cards = cards

    def snapshot(self) -> tuple:
        return self._cards


async def test_rep_floor_quantile_drops_bottom_of_bank_distribution(
    make_card, make_event
):
    # Bank: 4 harm-marked (lo20 ~0.045) + 4 cold (0.5); q=0.5 puts the floor at
    # the cold atom, so a harm slate card drops and a cold one survives.
    harm_bank = tuple(
        make_card(gain_events=tuple(make_event(-1.0) for _ in range(4)))
        for _ in range(4)
    )
    cold_bank = tuple(make_card() for _ in range(4))
    slate_harm, slate_cold = harm_bank[0], cold_bank[0]
    inner = StubShortlister(ResearchResult(cards=(slate_harm, slate_cold)))
    fused = FusedRankingShortlister(
        inner=inner,
        reputation=BetaBinomialReputation(),
        w_sem=1.0,
        w_rep=0.0,
        w_nov=0.0,
        rep_floor_quantile=0.5,
        store=StubStore(harm_bank + cold_bank),
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert [card.id for card in out.cards] == [slate_cold.id]


async def test_rep_floor_quantile_empty_bank_is_inert(make_card, make_event):
    card = make_card(gain_events=(make_event(-1.0),))
    inner = StubShortlister(ResearchResult(cards=(card,)))
    fused = FusedRankingShortlister(
        inner=inner,
        reputation=BetaBinomialReputation(),
        rep_floor_quantile=0.5,
        store=StubStore(()),
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert [c.id for c in out.cards] == [card.id]


async def test_rep_floor_quantile_excludes_benched_ids_upstream(make_card, make_event):
    # Harm bank cards sit below the q=0.5 floor (cold atom) AND have negative
    # magnitude: guaranteed downstream losers, so the fused gate must exclude
    # them from research (digest + index) instead of dropping them post-hoc.
    harm_bank = tuple(
        make_card(gain_events=tuple(make_event(-1.0) for _ in range(4)))
        for _ in range(4)
    )
    cold_bank = tuple(make_card() for _ in range(4))
    inner = StubShortlister(ResearchResult(cards=(cold_bank[0],)))
    fused = FusedRankingShortlister(
        inner=inner,
        reputation=BetaBinomialReputation(),
        w_sem=1.0,
        w_rep=0.0,
        w_nov=0.0,
        rep_floor_quantile=0.5,
        store=StubStore(harm_bank + cold_bank),
    )
    await fused.shortlist(**_shortlist_kwargs(), exclude_ids=frozenset({"lineage-x"}))
    sent = inner.calls[0]["exclude_ids"]
    assert "lineage-x" in sent
    assert {card.id for card in harm_bank} <= sent
    assert not ({card.id for card in cold_bank} & sent)


async def test_ev_dead_cards_excluded_upstream_even_above_rep_floor(
    make_card, make_event
):
    # `mixed` clears the rep floor (bank quantile lands on the deep-harm atom)
    # but its median gain is 0.0 <= 0: bid = theta x magnitude can never beat a
    # strictly positive EV reserve, so it is a guaranteed loser too.
    harm_bank = tuple(
        make_card(gain_events=tuple(make_event(-1.0) for _ in range(4)))
        for _ in range(4)
    )
    mixed = make_card(gain_events=(make_event(1.0), make_event(-1.0)))
    cold = make_card()
    inner = StubShortlister(ResearchResult(cards=(cold,)))
    fused = FusedRankingShortlister(
        inner=inner,
        reputation=BetaBinomialReputation(),
        w_sem=1.0,
        w_rep=0.0,
        w_nov=0.0,
        rep_floor_quantile=0.25,
        store=StubStore((*harm_bank, mixed, cold)),
    )
    await fused.shortlist(**_shortlist_kwargs())
    sent = inner.calls[0]["exclude_ids"]
    assert mixed.id in sent
    assert cold.id not in sent


async def test_bootstrap_rep_floor_keeps_cold_slate_card_below_warm_ev_floor(
    make_card, make_event
):
    warm_good = make_card(gain_events=(make_event(4.0),))
    warm_best = make_card(gain_events=(make_event(5.0),))
    cold = make_card()
    rep = BootstrapEVStubReputation({warm_good.id: 4.0, warm_best.id: 5.0})
    inner = StubShortlister(ResearchResult(cards=(cold,)))
    fused = BootstrapFusedRankingShortlister(
        inner=inner,
        reputation=rep,
        w_sem=1.0,
        w_rep=0.0,
        w_nov=0.0,
        rep_floor_quantile=0.5,
        store=StubStore((warm_good, warm_best, cold)),
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert [card.id for card in out.cards] == [cold.id]
    assert cold.id not in inner.calls[0]["exclude_ids"]


async def test_bootstrap_rep_floor_keeps_founding_only_card_explorable(
    make_card, make_event
):
    warm_good = make_card(gain_events=(make_event(4.0),))
    warm_best = make_card(gain_events=(make_event(5.0),))
    founding_only = make_card(gain_events=(make_event(0.1, founding=True),))
    rep = BootstrapEVStubReputation(
        {warm_good.id: 4.0, warm_best.id: 5.0, founding_only.id: 0.1}
    )
    inner = StubShortlister(ResearchResult(cards=(founding_only,)))
    fused = BootstrapFusedRankingShortlister(
        inner=inner,
        reputation=rep,
        w_sem=1.0,
        w_rep=0.0,
        w_nov=0.0,
        rep_floor_quantile=0.5,
        store=StubStore((warm_good, warm_best, founding_only)),
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert [card.id for card in out.cards] == [founding_only.id]
    assert founding_only.id not in inner.calls[0]["exclude_ids"]


async def test_bootstrap_single_unused_exposure_stays_explorable(make_card, make_event):
    # One ignored exposure is not proof of harm; benching at n=1 was the R1
    # absorbing death. Below the evidence floor the card stays researchable.
    unused_only = make_card(gain_events=(make_event(0.0, unused=True),))
    rep = BootstrapReputation(
        BetaBinomialReputation(), StubStore((unused_only,)), n_bootstrap=32
    )
    inner = StubShortlister(ResearchResult(cards=(unused_only,)))
    fused = BootstrapFusedRankingShortlister(
        inner=inner,
        reputation=rep,
        w_sem=1.0,
        w_rep=0.0,
        w_nov=0.0,
        rep_floor_quantile=0.4,
        store=StubStore((unused_only,)),
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert [card.id for card in out.cards] == [unused_only.id]
    assert unused_only.id not in inner.calls[0]["exclude_ids"]


async def test_bootstrap_confident_zero_support_is_benched(make_card, make_event):
    # Enough ignored exposure to clear the evidence floor with the optimistic
    # bootstrap EV read still non-positive: a proven no-op, benched upstream.
    ignored = make_card(
        gain_events=tuple(make_event(0.0, unused=True) for _ in range(3))
    )
    rep = BootstrapReputation(
        BetaBinomialReputation(), StubStore((ignored,)), n_bootstrap=32
    )
    inner = StubShortlister(ResearchResult(cards=(ignored,)))
    fused = BootstrapFusedRankingShortlister(
        inner=inner,
        reputation=rep,
        w_sem=1.0,
        w_rep=0.0,
        w_nov=0.0,
        rep_floor_quantile=0.4,
        store=StubStore((ignored,)),
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert out.cards == ()
    assert ignored.id in inner.calls[0]["exclude_ids"]


async def test_bootstrap_single_loss_is_not_benched(make_card, make_event):
    one_loss = make_card(gain_events=(make_event(-0.5),))
    rep = BootstrapReputation(
        BetaBinomialReputation(), StubStore((one_loss,)), n_bootstrap=32
    )
    inner = StubShortlister(ResearchResult(cards=(one_loss,)))
    fused = BootstrapFusedRankingShortlister(
        inner=inner,
        reputation=rep,
        w_sem=1.0,
        w_rep=0.0,
        w_nov=0.0,
        rep_floor_quantile=0.0,
        store=StubStore((one_loss,)),
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert [card.id for card in out.cards] == [one_loss.id]
    assert one_loss.id not in inner.calls[0]["exclude_ids"]


async def test_bootstrap_confident_loser_is_benched(make_card, make_event):
    loser = make_card(gain_events=tuple(make_event(-0.5) for _ in range(4)))
    rep = BootstrapReputation(
        BetaBinomialReputation(), StubStore((loser,)), n_bootstrap=32
    )
    inner = StubShortlister(ResearchResult(cards=(loser,)))
    fused = BootstrapFusedRankingShortlister(
        inner=inner,
        reputation=rep,
        w_sem=1.0,
        w_rep=0.0,
        w_nov=0.0,
        rep_floor_quantile=0.0,
        store=StubStore((loser,)),
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert out.cards == ()
    assert loser.id in inner.calls[0]["exclude_ids"]


@pytest.mark.parametrize(
    ("support_n", "expected_low_survives"),
    [(2, True), (3, False)],
    ids=["under-observed-exempt", "observed-dropped"],
)
async def test_bootstrap_rep_floor_respects_effective_support_boundary(
    make_card, make_event, support_n, expected_low_survives
):
    warm_low = make_card(gain_events=tuple(make_event(0.2) for _ in range(support_n)))
    warm_high = make_card(gain_events=(make_event(4.0),))
    rep = BootstrapEVStubReputation({warm_low.id: 0.2, warm_high.id: 4.0})
    inner = StubShortlister(ResearchResult(cards=(warm_low, warm_high)))
    fused = BootstrapFusedRankingShortlister(
        inner=inner,
        reputation=rep,
        w_sem=1.0,
        w_rep=0.0,
        w_nov=0.0,
        rep_floor_quantile=0.5,
        store=StubStore((warm_low, warm_high)),
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    expected = [warm_low.id, warm_high.id] if expected_low_survives else [warm_high.id]
    assert [card.id for card in out.cards] == expected
    assert warm_low.id not in inner.calls[0]["exclude_ids"]


def _old_event(gain: float, *, hours_ago: float) -> ContextualGain:
    return ContextualGain(
        context=DecisionContext(
            timestamp=datetime.now(UTC) - timedelta(hours=hours_ago)
        ),
        gain=gain,
        gain_se=0.0,
    )


async def test_bootstrap_bench_exempts_card_below_aged_effective_support(make_card):
    # Drift guard: a card whose raw loss COUNT clears the floor but whose
    # staleness-aged effective support is BELOW it must stay explorable. The
    # write evictor retains it (aged support < floor); the read bench must use
    # the SAME aged boundary, else the card is benched from auction AND probe
    # yet is unevictable — an absorbing zombie. Benching on the unaged
    # ``block.intro_events`` (which the bootstrap layer never re-ages) is the
    # bug: four old losses give raw count 4 >= 3 but aged support ~2.0 < 3.
    stale_loser = make_card(
        gain_events=tuple(_old_event(-0.5, hours_ago=1000.0) for _ in range(4))
    )
    fillers = tuple(
        make_card(gain_events=(_old_event(1.0, hours_ago=1.0),)) for _ in range(14)
    )
    bank = (stale_loser, *fillers)
    rep = BootstrapReputation(
        BetaBinomialReputation(),
        StubStore(bank),
        n_bootstrap=64,
        confident_min_events=3,
    )
    inner = StubShortlister(ResearchResult(cards=(stale_loser,)))
    fused = BootstrapFusedRankingShortlister(
        inner=inner,
        reputation=rep,
        w_sem=1.0,
        w_rep=0.0,
        w_nov=0.0,
        rep_floor_quantile=0.5,
        store=StubStore(bank),
    )
    out = await fused.shortlist(**_shortlist_kwargs())
    assert stale_loser.id not in inner.calls[0]["exclude_ids"]
    assert [card.id for card in out.cards] == [stale_loser.id]


async def test_upstream_exclusion_off_without_rep_floor_quantile(make_card, make_event):
    harm = make_card(gain_events=tuple(make_event(-1.0) for _ in range(4)))
    inner = StubShortlister(ResearchResult(cards=(harm,)))
    fused = FusedRankingShortlister(
        inner=inner,
        reputation=BetaBinomialReputation(),
        w_sem=0.0,
        w_rep=1.0,
        w_nov=0.0,
        score_floor=0.44,
    )
    await fused.shortlist(**_shortlist_kwargs(), exclude_ids=frozenset({"lineage-x"}))
    assert inner.calls[0]["exclude_ids"] == frozenset({"lineage-x"})


async def test_upstream_exclusion_never_changes_survivors(make_card, make_event):
    # Winner-set invariance at the shortlist level: pre-filtering the slate to
    # non-excluded cards yields the same survivor list as post-hoc dropping.
    harm_bank = tuple(
        make_card(gain_events=tuple(make_event(-1.0) for _ in range(4)))
        for _ in range(4)
    )
    cold_bank = tuple(make_card() for _ in range(4))
    bank = harm_bank + cold_bank

    def build(slate: tuple) -> FusedRankingShortlister:
        return FusedRankingShortlister(
            inner=StubShortlister(ResearchResult(cards=slate)),
            reputation=BetaBinomialReputation(),
            w_sem=1.0,
            w_rep=0.0,
            w_nov=0.0,
            rep_floor_quantile=0.5,
            store=StubStore(bank),
        )

    full_slate = (harm_bank[0], cold_bank[0], harm_bank[1], cold_bank[1])
    live_slate = (cold_bank[0], cold_bank[1])
    out_full = await build(full_slate).shortlist(**_shortlist_kwargs())
    out_live = await build(live_slate).shortlist(**_shortlist_kwargs())
    assert [c.id for c in out_full.cards] == [c.id for c in out_live.cards]


def test_rep_floor_quantile_requires_store():
    with pytest.raises(ValueError):
        FusedRankingShortlister(
            inner=StubShortlister(ResearchResult()),
            reputation=BetaBinomialReputation(),
            rep_floor_quantile=0.5,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"w_sem": -0.1},
        {"w_rep": -1.0},
        {"w_nov": -0.5},
        {"novelty_window_hours": 0.0},
        {"rep_floor_quantile": -0.1, "store": StubStore(())},
        {"rep_floor_quantile": 1.0, "store": StubStore(())},
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        FusedRankingShortlister(
            inner=StubShortlister(ResearchResult()),
            reputation=BetaBinomialReputation(),
            **kwargs,
        )


def test_full_yaml_ships_quantile_gated_read_path():
    cfg = OmegaConf.load(
        _REPO_ROOT / "config" / "memory" / "read_policy" / "adaptive.yaml"
    )
    shortlister = cfg.reader.shortlister
    assert (
        shortlister._target_
        == "gigaevo.memory.read.fused.BootstrapFusedRankingShortlister"
    )
    assert shortlister.w_sem == 1.0
    assert shortlister.w_rep == 0.0
    assert shortlister.w_nov == 0.0
    assert shortlister.rep_floor_quantile == 0.4
    raw = OmegaConf.to_container(shortlister, resolve=False)
    assert raw["store"] == "${ref:memory.store}"
    assert shortlister.get("score_floor") is None
    assert (
        shortlister.inner._target_
        == "gigaevo.memory.read.shortlist.ResearchShortlister"
    )


def test_portable_read_policy_ships_bootstrap_wrapper_with_lineage():
    cfg = OmegaConf.load(
        _REPO_ROOT / "config" / "memory" / "read_policy" / "portable.yaml"
    )
    shortlister = cfg.reader.shortlister
    assert (
        shortlister._target_
        == "gigaevo.memory.read.fused.BootstrapFusedRankingShortlister"
    )
    assert shortlister.w_rep == 0.0
    assert shortlister.w_nov == 0.0
    assert shortlister.get("score_floor") is None
    assert cfg.defaults[7]["/memory/excluder"] == "lineage"
    assert (
        shortlister.inner._target_
        == "gigaevo.memory.read.shortlist.ResearchShortlister"
    )
