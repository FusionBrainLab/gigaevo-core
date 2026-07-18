"""Shortlisting: research-query assembly and the store-backed shortlister."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.memory.cards import ContextualGain, DecisionContext
from gigaevo.memory.read.shortlist import (
    ResearchShortlister,
    _bank_digest,
    build_research_query,
)
from gigaevo.memory.storage.base import ResearchFailure, ResearchRequest, ResearchResult


def _stamped(ts: datetime | None) -> ContextualGain:
    return ContextualGain(context=DecisionContext(timestamp=ts), gain=0.1)


class _Parent:
    def __init__(self, code: str, context: str = "") -> None:
        self.code = code
        self.metadata = {MUTATION_CONTEXT_METADATA_KEY: context} if context else {}


class _RecordingStore:
    def __init__(self, result: ResearchResult | None = None, bank: tuple = ()) -> None:
        self.requests: list[ResearchRequest] = []
        self._result = result or ResearchResult()
        self._bank = bank

    def snapshot(self) -> tuple:
        return self._bank

    async def research(self, request: ResearchRequest) -> ResearchResult:
        self.requests.append(request)
        return self._result


class _ExplodingStore:
    def snapshot(self) -> tuple:
        return ()

    async def research(self, request: ResearchRequest) -> ResearchResult:
        raise RuntimeError("store down")


class _SnapshotExplodingStore(_RecordingStore):
    def snapshot(self) -> tuple:
        raise RuntimeError("bank unreadable")


class TestBuildResearchQuery:
    def test_contains_all_mutation_inputs(self):
        query = build_research_query(
            parents=[_Parent("def f(): pass", context="snapshot A")],
            mutation_mode="crossover",
            task_description="pack points",
            metrics_description="fitness: min distance",
        )
        assert "TASK DESCRIPTION:\npack points" in query
        assert "AVAILABLE METRICS:\nfitness: min distance" in query
        assert "MUTATION MODE:\ncrossover" in query
        assert "=== Parent 1 ===" in query
        assert "def f(): pass" in query
        assert "snapshot A" in query
        assert query.endswith(
            "transferability, and failure risk. Return no card if none clears that bar."
        )

    def test_empty_fields_get_placeholders(self):
        query = build_research_query(
            parents=[],
            mutation_mode="",
            task_description="  ",
            metrics_description="",
        )
        assert "TASK DESCRIPTION:\n<empty>" in query
        assert "AVAILABLE METRICS:\n<empty>" in query
        assert "MUTATION MODE:\nrewrite" in query

    def test_explicit_parent_contexts_override_metadata(self):
        query = build_research_query(
            parents=[_Parent("code0", context="stale"), _Parent("code1")],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
            parent_contexts=["fresh ctx", ""],
        )
        assert "fresh ctx" in query
        assert "stale" not in query
        assert "=== Parent 2 ===" in query

    def test_metadata_context_used_when_no_override(self):
        query = build_research_query(
            parents=[_Parent("code0", context="from metadata")],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
        )
        assert "from metadata" in query


class TestResearchShortlister:
    @pytest.mark.asyncio
    async def test_threads_query_and_exclusions_to_store(self, make_card):
        card = make_card()
        store = _RecordingStore(ResearchResult(cards=(card,), iterations=2))
        shortlister = ResearchShortlister(store)
        result = await shortlister.shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
            exclude_ids=frozenset({"m-old"}),
        )
        assert result.cards == (card,)
        (request,) = store.requests
        assert request.exclude_ids == frozenset({"m-old"})
        assert "MUTATION INPUTS" in request.query

    @pytest.mark.asyncio
    async def test_store_failure_preserves_a_neutral_failure_result(self):
        shortlister = ResearchShortlister(_ExplodingStore())
        result = await shortlister.shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
        )
        assert result.failure is ResearchFailure.STORE_EXCEPTION

    @pytest.mark.asyncio
    async def test_planning_context_carries_bank_digest(self, make_card):
        bank = (
            make_card(description="anneal the radius schedule"),
            make_card(description="cache pairwise distances"),
        )
        store = _RecordingStore(bank=bank)
        await ResearchShortlister(store).shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
        )
        (request,) = store.requests
        assert "BANK CONTENTS (2 cards)" in request.planning_context
        assert "anneal the radius schedule" in request.planning_context
        assert "cache pairwise distances" in request.planning_context
        assert "BANK CONTENTS" not in request.query

    @pytest.mark.asyncio
    async def test_digest_omits_excluded_cards(self, make_card):
        kept = make_card(description="anneal the radius schedule")
        dropped = make_card(description="cache pairwise distances")
        store = _RecordingStore(bank=(kept, dropped))
        await ResearchShortlister(store).shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
            exclude_ids=frozenset({dropped.id}),
        )
        (request,) = store.requests
        assert "BANK CONTENTS (1 card)" in request.planning_context
        assert "anneal the radius schedule" in request.planning_context
        assert "cache pairwise distances" not in request.planning_context

    @pytest.mark.asyncio
    async def test_exclusion_expands_absorbed_aliases(self, make_card):
        survivor = make_card(
            id="mem-new",
            absorbed_ids=("mem-old",),
            description="merged old lever",
        )
        store = _RecordingStore(bank=(survivor,))
        await ResearchShortlister(store).shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
            exclude_ids=frozenset({"mem-old"}),
        )
        (request,) = store.requests
        assert request.exclude_ids == frozenset({"mem-old", "mem-new"})
        assert "merged old lever" not in request.planning_context

    @pytest.mark.asyncio
    async def test_excluded_cards_free_digest_cap_slots(self, make_card):
        stamped = tuple(
            make_card(
                description=f"stamped {n}",
                gain_events=(_stamped(datetime(2026, 1, 1, n + 1, tzinfo=UTC)),),
            )
            for n in range(3)
        )
        cold = make_card(description="cold newcomer")
        store = _RecordingStore(bank=(*stamped, cold))
        await ResearchShortlister(store, digest_max_cards=3).shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
            exclude_ids=frozenset({stamped[0].id}),
        )
        (request,) = store.requests
        assert "cold newcomer" in request.planning_context
        assert "stamped 0" not in request.planning_context

    @pytest.mark.asyncio
    async def test_bank_digest_caps_at_max_cards_and_marks_dropped(self, make_card):
        bank = tuple(
            make_card(description=f"card {n}: " + "x" * 150) for n in range(100)
        )
        store = _RecordingStore(bank=bank)
        await ResearchShortlister(store).shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
        )
        (request,) = store.requests
        digest = request.planning_context
        assert "BANK CONTENTS (100 cards)" in digest
        assert digest.count("\n- ") == 50
        assert "(+50 more cards not shown)" in digest

    @pytest.mark.asyncio
    async def test_digest_max_cards_is_configurable(self, make_card):
        bank = tuple(make_card(description=f"idea {n}") for n in range(8))
        store = _RecordingStore(bank=bank)
        await ResearchShortlister(store, digest_max_cards=3).shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
        )
        (request,) = store.requests
        assert request.planning_context.count("\n- ") == 3
        assert "(+5 more cards not shown)" in request.planning_context

    def test_rejects_nonpositive_digest_max_cards(self):
        with pytest.raises(ValueError):
            ResearchShortlister(_RecordingStore(), digest_max_cards=0)
        with pytest.raises(ValueError):
            ResearchShortlister(_RecordingStore(), digest_max_cards=-1)

    @pytest.mark.asyncio
    async def test_empty_bank_yields_empty_planning_context(self):
        store = _RecordingStore()
        await ResearchShortlister(store).shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
        )
        (request,) = store.requests
        assert request.planning_context == ""

    @pytest.mark.asyncio
    async def test_digest_lists_newest_cards_first(self, make_card):
        bank = (
            make_card(
                description="old idea",
                gain_events=(_stamped(datetime(2026, 1, 1, tzinfo=UTC)),),
            ),
            make_card(
                description="mid idea",
                gain_events=(
                    _stamped(datetime(2026, 1, 15, tzinfo=UTC)),
                    _stamped(datetime(2026, 2, 1, tzinfo=UTC)),
                ),
            ),
            make_card(
                description="new idea",
                gain_events=(_stamped(datetime(2026, 3, 1, tzinfo=UTC)),),
            ),
        )
        store = _RecordingStore(bank=bank)
        await ResearchShortlister(store).shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
        )
        (request,) = store.requests
        digest = request.planning_context
        assert (
            digest.index("new idea")
            < digest.index("mid idea")
            < digest.index("old idea")
        )

    @pytest.mark.asyncio
    async def test_snapshot_failure_still_shortlists(self, make_card):
        card = make_card()
        store = _SnapshotExplodingStore(ResearchResult(cards=(card,)))
        result = await ResearchShortlister(store).shortlist(
            parents=[_Parent("code")],
            mutation_mode="rewrite",
            task_description="task",
            metrics_description="metrics",
        )
        assert result.cards == (card,)
        (request,) = store.requests
        assert request.planning_context == ""


class TestBankDigest:
    def test_unstamped_cards_sort_after_stamped_in_creation_order(self, make_card):
        none_ts = make_card(description="none-ts idea", gain_events=(_stamped(None),))
        eventless = make_card(description="eventless idea")
        dated = make_card(
            description="dated idea",
            gain_events=(_stamped(datetime(2026, 1, 1, tzinfo=UTC)),),
        )
        digest = _bank_digest((none_ts, eventless, dated), max_cards=10)
        assert (
            digest.index("dated idea")
            < digest.index("none-ts idea")
            < digest.index("eventless idea")
        )

    def test_naive_timestamps_read_as_utc(self, make_card):
        aware_older = make_card(
            description="aware older",
            gain_events=(_stamped(datetime(2026, 5, 1, 11, 0, tzinfo=UTC)),),
        )
        naive_newer = make_card(
            description="naive newer",
            gain_events=(_stamped(datetime(2026, 5, 1, 12, 0)),),
        )
        digest = _bank_digest((aware_older, naive_newer), max_cards=10)
        assert digest.index("naive newer") < digest.index("aware older")

    def test_mixed_naive_and_aware_stamps_on_one_card(self, make_card):
        # max() over mixed aware/naive stamps raises TypeError unless naive
        # stamps are normalized to UTC first — this pins the normalization on
        # any host timezone, where the two-card test above only discriminates
        # on non-UTC hosts.
        mixed = make_card(
            description="mixed stamps",
            gain_events=(
                _stamped(datetime(2026, 5, 1, 12, 0)),
                _stamped(datetime(2026, 5, 1, 11, 0, tzinfo=UTC)),
            ),
        )
        aware = make_card(
            description="aware only",
            gain_events=(_stamped(datetime(2026, 5, 1, 11, 30, tzinfo=UTC)),),
        )
        digest = _bank_digest((aware, mixed), max_cards=10)
        assert digest.index("mixed stamps") < digest.index("aware only")

    def test_timestamp_ties_break_on_card_id(self, make_card):
        ts = datetime(2026, 4, 1, tzinfo=UTC)
        low_id = make_card(description="low id idea", gain_events=(_stamped(ts),))
        high_id = make_card(description="high id idea", gain_events=(_stamped(ts),))
        digest = _bank_digest((high_id, low_id), max_cards=10)
        assert digest.index("low id idea") < digest.index("high id idea")

    def test_only_description_is_rendered(self, make_card):
        card = make_card(
            description="anneal radius",
            explanation_summary="supporting mechanism",
        )
        digest = _bank_digest((card,), max_cards=10)
        assert "- anneal radius" in digest.splitlines()
        assert "supporting mechanism" not in digest

    def test_long_description_ellipsized(self, make_card):
        card = make_card(description="y" * 205)
        digest = _bank_digest((card,), max_cards=10)
        assert "- " + "y" * 199 + "…" in digest.splitlines()

    def test_whitespace_only_description_yields_bare_line(self, make_card):
        card = make_card(description="   \n  ")
        digest = _bank_digest((card,), max_cards=10)
        assert "BANK CONTENTS (1 card):" in digest
        assert "- " in digest.splitlines()

    def test_plural_bank_header(self, make_card):
        bank = (make_card(description="a"), make_card(description="b"))
        digest = _bank_digest(bank, max_cards=10)
        assert "BANK CONTENTS (2 cards):" in digest

    def test_exactly_max_cards_shows_all_without_marker(self, make_card):
        bank = tuple(make_card(description=f"idea {n}") for n in range(5))
        digest = _bank_digest(bank, max_cards=5)
        assert digest.count("\n- ") == 5
        assert "not shown" not in digest

    def test_one_over_max_cards_drops_the_oldest(self, make_card):
        newest = make_card(
            description="newest idea",
            gain_events=(_stamped(datetime(2026, 6, 1, tzinfo=UTC)),),
        )
        oldest = make_card(
            description="oldest idea",
            gain_events=(_stamped(datetime(2026, 1, 1, tzinfo=UTC)),),
        )
        digest = _bank_digest((oldest, newest), max_cards=1)
        assert "newest idea" in digest
        assert "oldest idea" not in digest
        assert "(+1 more card not shown)" in digest
