"""Tests for ParentRefresher and ParentRefreshSelector."""

from __future__ import annotations

import asyncio
import contextlib
import gc

import pytest

from gigaevo.evolution.engine.refresh import (
    CoalescedRefreshResult,
    DirectParentsSelector,
    ParentRefresher,
    ParentRefreshSelector,
    ParentRefreshTicket,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.evolution._fake_dag import FakeDag, build_test_refresher


@pytest.mark.asyncio
async def test_refresh_single_parent_round_trip(fakeredis_storage):
    """A single DONE parent is flipped to QUEUED and re-awaited to DONE."""
    refresher, parent, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        refreshed = await refresher.refresh([parent])
        assert len(refreshed) == 1
        assert refreshed[0].id == parent.id
        assert refreshed[0].state == ProgramState.DONE
        assert fake_dag.evaluations == 1
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_empty_list_returns_empty(fakeredis_storage):
    """Empty parents list short-circuits to empty result."""
    refresher, _, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        result = await refresher.refresh([])
        assert result == []
        # FakeDag's seed program counts in flip_count_for if it ever flipped;
        # an empty refresh should not have evaluated anything.
        assert fake_dag.evaluations == 0
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_two_parents_batch(fakeredis_storage):
    """Two parents are flipped together and awaited as a batch."""
    refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        p2 = await fake_dag.add_program("p2")
        refreshed = await refresher.refresh([p1, p2])
        refreshed_ids = {p.id for p in refreshed}
        assert refreshed_ids == {p1.id, p2.id}
        assert fake_dag.evaluations == 2
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_overlapping_parents_serialised(fakeredis_storage):
    """Two concurrent refresh() calls sharing one parent do not double-flip it."""
    refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        p2 = await fake_dag.add_program("p2")
        p3 = await fake_dag.add_program("p3")

        # Concurrently refresh overlapping sets. p1 is in both; the per-id
        # lock must serialise the two refreshes so p1 flips exactly twice
        # total (once per refresh, never concurrently).
        a, b = await asyncio.gather(
            refresher.refresh([p1, p2]),
            refresher.refresh([p1, p3]),
        )

        a_ids = {p.id for p in a}
        b_ids = {p.id for p in b}
        assert p1.id in a_ids and p1.id in b_ids
        # p1 was flipped twice — once per serialised refresh.
        assert fake_dag.flip_count_for(p1.id) == 2
        assert fake_dag.flip_count_for(p2.id) == 1
        assert fake_dag.flip_count_for(p3.id) == 1
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_discarded_parent_raises(fakeredis_storage):
    """A DISCARDED parent passed in raises rather than flipping it."""
    refresher, parent, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        # Mutate the in-memory copy directly — caller passes a Program whose
        # state field is DISCARDED, simulating a stale producer view.
        parent.state = ProgramState.DISCARDED
        with pytest.raises(ValueError, match="DISCARDED"):
            await refresher.refresh([parent])
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_timeout_when_dag_frozen(fakeredis_storage):
    """If the DAG never finishes the refresh, the refresher times out."""
    refresher, parent, fake_dag = await build_test_refresher(fakeredis_storage)
    refresher._timeout_seconds = 0.2  # tight bound for this test
    try:
        fake_dag.frozen_ids.add(parent.id)
        with pytest.raises(TimeoutError, match="timed out"):
            await refresher.refresh([parent])
    finally:
        fake_dag.frozen_ids.discard(parent.id)
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_default_timeout_absorbs_brief_pause(fakeredis_storage):
    """Default ``timeout_seconds=600.0`` is far longer than any healthy
    refresh, so a brief DAG pause is absorbed and the refresh still
    completes successfully without bumping into the bound."""
    refresher, parent, fake_dag = await build_test_refresher(fakeredis_storage)
    # H3 mitigation: default is finite (10 min) so a DAG-runner crash
    # cannot strand a mutant task forever. Healthy refreshes finish
    # in milliseconds, well under this bound.
    assert refresher._timeout_seconds == 600.0
    try:
        # Hold the parent QUEUED for ~150ms — longer than the poll interval
        # but still short enough that the test runs fast. With no timeout
        # configured, the refresher must wait through the pause rather
        # than raise.
        fake_dag.frozen_ids.add(parent.id)

        async def _release_soon():
            await asyncio.sleep(0.15)
            fake_dag.frozen_ids.discard(parent.id)

        releaser = asyncio.create_task(_release_soon())
        try:
            refreshed = await asyncio.wait_for(refresher.refresh([parent]), timeout=5.0)
            assert len(refreshed) == 1
            assert refreshed[0].state == ProgramState.DONE
        finally:
            await releaser
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_parent_becomes_discarded_midflight(fakeredis_storage):
    """A parent flipped to DISCARDED by another path *during* refresh raises
    ``ValueError`` rather than returning stale state."""
    refresher, parent, fake_dag = await build_test_refresher(fakeredis_storage)
    refresher._poll_interval = 0.01
    try:
        # Freeze the DAG so the parent stays QUEUED, then race a DISCARD
        # transition against the refresher's poll loop.
        fake_dag.frozen_ids.add(parent.id)

        async def _discard_after_flip():
            # Wait for the refresher to flip DONE→QUEUED, then bypass the
            # DAG and force the parent to DISCARDED.
            for _ in range(200):
                cur = await fakeredis_storage.get(parent.id)
                if cur is not None and cur.state == ProgramState.QUEUED:
                    break
                await asyncio.sleep(0.005)
            await fakeredis_storage.batch_transition_by_ids(
                [parent.id],
                ProgramState.QUEUED.value,
                ProgramState.DISCARDED.value,
            )

        discarder = asyncio.create_task(_discard_after_flip())
        try:
            with pytest.raises(ValueError, match="DISCARDED"):
                await asyncio.wait_for(refresher.refresh([parent]), timeout=5.0)
        finally:
            await discarder
    finally:
        fake_dag.frozen_ids.discard(parent.id)
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_parent_vanishes_midflight(fakeredis_storage):
    """A parent removed from storage *during* refresh raises ``ValueError``."""
    refresher, parent, fake_dag = await build_test_refresher(fakeredis_storage)
    refresher._poll_interval = 0.01
    try:
        fake_dag.frozen_ids.add(parent.id)

        async def _remove_after_flip():
            for _ in range(200):
                cur = await fakeredis_storage.get(parent.id)
                if cur is not None and cur.state == ProgramState.QUEUED:
                    break
                await asyncio.sleep(0.005)
            await fakeredis_storage.remove(parent.id)

        remover = asyncio.create_task(_remove_after_flip())
        try:
            with pytest.raises(ValueError, match="vanished"):
                await asyncio.wait_for(refresher.refresh([parent]), timeout=5.0)
        finally:
            await remover
    finally:
        fake_dag.frozen_ids.discard(parent.id)
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_reversed_input_order_does_not_deadlock(fakeredis_storage):
    """Two concurrent refreshes with reversed input ordering of the same
    parent set must both complete — the per-id lock acquisition order is
    sorted by id, so both callers grab locks in the same global order and
    cannot deadlock against each other."""
    refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        p2 = await fake_dag.add_program("p2")
        p3 = await fake_dag.add_program("p3")

        # A passes [p1,p2,p3], B passes [p3,p2,p1]. Without deterministic
        # sorting, classic lock-order inversion would deadlock under
        # contention; the refresher sorts by id so both must succeed.
        a, b = await asyncio.wait_for(
            asyncio.gather(
                refresher.refresh([p1, p2, p3]),
                refresher.refresh([p3, p2, p1]),
            ),
            timeout=5.0,
        )
        assert {p.id for p in a} == {p1.id, p2.id, p3.id}
        assert {p.id for p in b} == {p1.id, p2.id, p3.id}
        # Each parent flips exactly twice — once per serialised refresh.
        for p in (p1, p2, p3):
            assert fake_dag.flip_count_for(p.id) == 2
    finally:
        await fake_dag.stop()


# ---------------------------------------------------------------------------
# ParentRefreshSelector ABC + DirectParentsSelector default
# ---------------------------------------------------------------------------


class TestDirectParentsSelector:
    @pytest.mark.asyncio
    async def test_returns_input_unchanged(self):
        selector = DirectParentsSelector()
        p1 = Program(code="def a(): pass")
        p2 = Program(code="def b(): pass")
        result = await selector.select([p1, p2])
        assert result == [p1, p2]

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        selector = DirectParentsSelector()
        assert await selector.select([]) == []

    def test_is_parent_refresh_selector_subclass(self):
        assert issubclass(DirectParentsSelector, ParentRefreshSelector)


class TestRefresherUsesSelector:
    @pytest.mark.asyncio
    async def test_custom_selector_can_add_targets(self, fakeredis_storage):
        """A selector that adds extra targets routes them through refresh too."""
        fake_dag = FakeDag(fakeredis_storage)
        fake_dag.start()
        try:
            p1 = await fake_dag.add_program("p1")
            p2 = await fake_dag.add_program("p2")

            class AddP2Selector(ParentRefreshSelector):
                async def select(self, parents):
                    return [*parents, p2]

            refresher = ParentRefresher(
                storage=fakeredis_storage,
                selector=AddP2Selector(),
                poll_interval=0.02,
            )
            refreshed = await refresher.refresh([p1])
            refreshed_ids = {p.id for p in refreshed}
            assert refreshed_ids == {p1.id, p2.id}
            assert fake_dag.evaluations == 2
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_empty_selector_short_circuits(self, fakeredis_storage):
        """If the selector returns nothing, refresh() is a no-op."""
        fake_dag = FakeDag(fakeredis_storage)
        fake_dag.start()
        try:
            p1 = await fake_dag.add_program("p1")

            class EmptySelector(ParentRefreshSelector):
                async def select(self, parents):
                    return []

            refresher = ParentRefresher(
                storage=fakeredis_storage,
                selector=EmptySelector(),
                poll_interval=0.02,
            )
            result = await refresher.refresh([p1])
            assert result == []
            assert fake_dag.evaluations == 0
        finally:
            await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_does_not_deadlock_on_duplicate_parent_ids(fakeredis_storage):
    """``asyncio.Lock`` is not reentrant, so a duplicate program id in
    the parents list would make ``_acquire_all`` call ``acquire()``
    twice on the same Lock and hang the mutant task forever, holding
    its in-flight slot. The refresher must dedupe before lock
    acquisition."""
    refresher, parent, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        # Duplicate the parent. A non-reentrant-Lock bug would hang here
        # rather than completing under the wait_for budget.
        refreshed = await asyncio.wait_for(
            refresher.refresh([parent, parent, parent]),
            timeout=5.0,
        )
        # Result must collapse to a single program (deduped).
        assert len(refreshed) == 1
        assert refreshed[0].id == parent.id
        assert refreshed[0].state == ProgramState.DONE
        # The DAG must have evaluated the parent exactly once — duplicates
        # are folded out before the batch transition, not flipped N times.
        assert fake_dag.flip_count_for(parent.id) == 1
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_selector_emitting_duplicates_does_not_deadlock(
    fakeredis_storage,
):
    """A custom selector that returns the same parent twice must not
    deadlock either — dedup happens after selector.select(), not before."""
    fake_dag = FakeDag(fakeredis_storage)
    fake_dag.start()
    try:
        p1 = await fake_dag.add_program("p1")

        class DupSelector(ParentRefreshSelector):
            async def select(self, parents):
                # Returns the same parent twice via a "lineage walk" that
                # accidentally visits the same id more than once.
                return [*parents, *parents]

        refresher = ParentRefresher(
            storage=fakeredis_storage,
            selector=DupSelector(),
            poll_interval=0.02,
        )
        refreshed = await asyncio.wait_for(
            refresher.refresh([p1]),
            timeout=5.0,
        )
        assert len(refreshed) == 1
        assert refreshed[0].id == p1.id
        assert fake_dag.flip_count_for(p1.id) == 1
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_locks_dict_does_not_grow_unboundedly(fakeredis_storage):
    """A strong-ref ``_locks`` registry would retain every parent-id
    lock ever created, leaking ~1 lock per distinct parent on multi-day
    runs. With ``WeakValueDictionary``, locks are released for GC once
    their refresh completes."""
    fake_dag = FakeDag(fakeredis_storage)
    fake_dag.start()
    try:
        # Distinct parents per refresh so each forces a new lock entry.
        parents = []
        for i in range(20):
            p = await fake_dag.add_program(f"p{i}")
            parents.append(p)

        refresher = ParentRefresher(
            storage=fakeredis_storage,
            poll_interval=0.02,
        )
        # Sequentially refresh each parent. Each call holds a strong ref
        # to one lock during the call; the dict entry is reclaimable
        # once the call returns and the local refs are gone.
        for p in parents:
            await refresher.refresh([p])

        # Force a collection so the WeakValueDictionary drops dead entries.
        gc.collect()
        # All 20 lock entries should be gone — no caller holds references.
        # We tolerate at most a handful that linger due to event-loop
        # internals retaining transient references.
        assert len(refresher._locks) <= 2, (
            f"locks dict did not release entries: {len(refresher._locks)} remain"
        )
    finally:
        await fake_dag.stop()


# ---------------------------------------------------------------------------
# Deadlock stress
#
# These tests assault the per-id-locked refresh pipeline with the
# failure modes that produce deadlocks in lock-fan-in / lock-fan-out
# designs:
#
#   1. N-way contention on a single resource (same parent, many callers).
#   2. ABBA ordering risk across overlapping batches (random input order
#      per caller, identical target sets, sorted-acquire is the defence).
#   3. Cancellation interleaved with lock acquisition + critical
#      section (flip, await DAG, release).
#   4. Distinct-parent churn at scale (registry must stay bounded under
#      WeakValueDictionary GC pressure).
#
# All tests wrap their work in ``asyncio.wait_for`` with a tight wall-
# clock budget — if anything deadlocks, the test fails fast with a
# TimeoutError rather than hanging the suite.
# ---------------------------------------------------------------------------


_DEADLOCK_BUDGET_S = 10.0


@pytest.mark.asyncio
async def test_refresh_32way_same_parent_storm_no_deadlock(fakeredis_storage):
    """32 concurrent ``refresh([X])`` calls on a single parent.

    Each caller serialises on X's lock, so we expect exactly 32 sequential
    flips of X. The risk being tested: a caller crashing mid-acquire,
    or the lock leaking, would either deadlock the remaining callers
    or produce fewer than 32 flips. Both fail fast under ``wait_for``.
    """
    refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[refresher.refresh([p1]) for _ in range(32)]),
            timeout=_DEADLOCK_BUDGET_S,
        )
        # All 32 succeeded with the same parent id.
        assert len(results) == 32
        for r in results:
            assert len(r) == 1 and r[0].id == p1.id
        # Each caller forces its own flip — duplicates aren't coalesced.
        # If the lock isn't actually serialising, flips would be < 32
        # (some callers would race past the lock and find p1 already
        # mid-flight, but the refresher does NOT currently coalesce, so
        # the contract is "32 sequential flips, no deadlock").
        assert fake_dag.flip_count_for(p1.id) == 32, (
            f"expected 32 flips, got {fake_dag.flip_count_for(p1.id)} — "
            "lock not serialising callers"
        )
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_random_order_overlapping_batches_no_deadlock(
    fakeredis_storage,
):
    """16 concurrent refreshers on overlapping parent sets with randomised
    input order — classic ABBA-deadlock setup.

    Each caller picks a random permutation of [p1..p4]. Without the
    sorted-by-id lock acquisition order in ``ParentRefresher.refresh``,
    pairs of callers would deadlock: A holds p1 waiting for p2 while
    B holds p2 waiting for p1. With sorted acquire, all callers grab
    locks in the same global order — deadlock is impossible.
    """
    import random

    refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        p2 = await fake_dag.add_program("p2")
        p3 = await fake_dag.add_program("p3")
        p4 = await fake_dag.add_program("p4")
        parents = [p1, p2, p3, p4]

        rng = random.Random(42)  # deterministic permutations
        permutations = [rng.sample(parents, k=len(parents)) for _ in range(16)]

        results = await asyncio.wait_for(
            asyncio.gather(*[refresher.refresh(perm) for perm in permutations]),
            timeout=_DEADLOCK_BUDGET_S,
        )
        target_ids = {p.id for p in parents}
        for r in results:
            assert {p.id for p in r} == target_ids, (
                "refresh dropped or duplicated targets under contention"
            )
        # Each parent flipped exactly 16 times — once per caller. If any
        # pair deadlocked, asyncio.gather + wait_for would have raised
        # TimeoutError above.
        for p in parents:
            assert fake_dag.flip_count_for(p.id) == 16
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_cancel_releases_locks_for_waiters(fakeredis_storage):
    """A caller cancelled mid-flight must release its per-id locks so
    waiting callers can make progress.

    Pattern:
      1. Block the DAG by freezing p1's id — caller A's refresh stalls
         inside ``_await_done`` polling.
      2. Caller B starts a second refresh of the same parent — it queues
         on A's lock.
      3. Cancel A. The ``_acquire_all`` context manager's ``finally``
         must release A's lock so B proceeds.
      4. Unfreeze p1 → B's refresh completes.

    If ``_acquire_all`` or the surrounding async-with mishandled cancel,
    B would block indefinitely. We bound the wait with ``wait_for``.
    """
    refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        fake_dag.frozen_ids.add(p1.id)
        a_task = asyncio.create_task(refresher.refresh([p1]))
        # Let A enter the lock + start polling (otherwise we'd race the
        # lock acquisition itself).
        await asyncio.sleep(0.05)

        b_task = asyncio.create_task(refresher.refresh([p1]))
        # B should be blocked on A's lock — give it a moment to confirm.
        await asyncio.sleep(0.05)
        assert not b_task.done(), "B should be waiting on A's lock"

        # Cancel A. Its finally must release the lock for B.
        a_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await a_task

        # Unfreeze so B's DAG poll can complete.
        fake_dag.frozen_ids.discard(p1.id)

        b_result = await asyncio.wait_for(b_task, timeout=_DEADLOCK_BUDGET_S)
        assert len(b_result) == 1 and b_result[0].id == p1.id
    finally:
        fake_dag.frozen_ids.clear()
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_cancel_mid_acquire_does_not_strand_partial_locks(
    fakeredis_storage,
):
    """Caller cancelled mid lock-acquisition loop must release the
    locks it already held — otherwise a later caller for the same id
    would deadlock against the stranded lock.

    We pin parents [p1..p3] in sorted-id order. A acquires p1, p2 then
    blocks on p3 (held by a synthetic blocker). Cancelling A must
    release p1 and p2 so a parallel B can proceed on those ids.
    """
    refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        p2 = await fake_dag.add_program("p2")
        p3 = await fake_dag.add_program("p3")

        # Manually grab p3's lock to force A to block on it. Reach into
        # the registry the same way `_get_lock` does so we hold the
        # exact same Lock instance the refresher would.
        p3_lock = await refresher._get_lock(p3.id)
        await p3_lock.acquire()

        a_task = asyncio.create_task(refresher.refresh([p1, p2, p3]))
        # Give A time to acquire p1 + p2 and then block on p3.
        await asyncio.sleep(0.1)
        assert not a_task.done()

        # Cancel A. The finally inside _acquire_all must release the
        # locks A holds (p1, p2).
        a_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await a_task

        # Now a B touching p1+p2 must NOT deadlock.
        b_result = await asyncio.wait_for(
            refresher.refresh([p1, p2]), timeout=_DEADLOCK_BUDGET_S
        )
        assert {p.id for p in b_result} == {p1.id, p2.id}

        # Release the synthetic blocker so the test cleans up.
        p3_lock.release()
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_high_churn_keeps_registry_bounded(fakeredis_storage):
    """Under heavy churn across hundreds of distinct parents, the
    ``_locks`` registry must stay near-empty between refreshes thanks
    to the WeakValueDictionary's GC of dead entries.

    Failure mode being guarded: a strong reference somewhere in the
    refresh pipeline (e.g. captured in a closure or accumulated in a
    list) would defeat the weak-value semantics and the dict would
    grow unbounded on multi-day runs touching tens of thousands of
    distinct parents.
    """
    fake_dag = FakeDag(fakeredis_storage)
    fake_dag.start()
    try:
        refresher = ParentRefresher(storage=fakeredis_storage, poll_interval=0.01)

        # 200 distinct parents in 20 batches of 10. Each batch fully
        # completes before the next starts → no caller holds a strong
        # ref to old batch's locks once gc kicks in.
        parents_per_batch = 10
        n_batches = 20
        for batch_idx in range(n_batches):
            batch = [
                await fake_dag.add_program(f"churn_b{batch_idx}_i{i}")
                for i in range(parents_per_batch)
            ]
            await refresher.refresh(batch)
            del batch  # drop strong refs from this frame
            gc.collect()

        # After 200 unique parents and full GC, the registry should be
        # essentially empty (tolerate a couple of transient lingerers
        # from event-loop bookkeeping).
        assert len(refresher._locks) <= 4, (
            f"locks registry leaked under churn: {len(refresher._locks)} "
            f"entries remain after {n_batches * parents_per_batch} unique "
            "parents — strong reference leak suspected"
        )
    finally:
        await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_cancel_during_await_done_no_lock_leak(fakeredis_storage):
    """Caller cancelled while inside ``_await_done`` polling must
    release its lock so subsequent callers complete cleanly.

    This exercises a different code path from the earlier
    cancel-mid-acquire test: here the lock IS held, the flip HAS landed,
    and the cancel arrives during the DONE-poll loop. The
    ``async with _acquire_all`` block's finally is the only thing
    preventing a permanent lock leak.
    """
    refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        # Freeze so the poll loop spins indefinitely.
        fake_dag.frozen_ids.add(p1.id)
        a_task = asyncio.create_task(refresher.refresh([p1]))
        # Wait for A to enter the poll loop (flip already landed).
        await asyncio.sleep(0.1)
        a_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await a_task

        # Unfreeze. Now a fresh caller B must proceed without waiting on
        # A's lock indefinitely.
        fake_dag.frozen_ids.discard(p1.id)
        b_result = await asyncio.wait_for(
            refresher.refresh([p1]), timeout=_DEADLOCK_BUDGET_S
        )
        assert len(b_result) == 1 and b_result[0].id == p1.id
    finally:
        fake_dag.frozen_ids.clear()
        await fake_dag.stop()


# ---------------------------------------------------------------------------
# refresh_with_ticket — extended-ownership API used by mutant_task to hold
# per-parent-id locks past refresh and through the child's DAG evaluation.
#
# The contract:
#   * refresh_with_ticket returns a ParentRefreshTicket holding the parent
#     locks. The locks are NOT released when the call returns.
#   * The caller MUST call ticket.release() exactly once when ownership
#     ends (after the child mutant has been ingested or discarded).
#   * release() is idempotent.
#   * While the ticket is held, a concurrent refresh()/refresh_with_ticket()
#     call for the same parent BLOCKS until the ticket releases.
#
# These tests are the failing-RED step before the implementation lands.
# ---------------------------------------------------------------------------


class TestRefreshWithTicket:
    @pytest.mark.asyncio
    async def test_returns_ticket_with_refreshed_programs(self, fakeredis_storage):
        refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            ticket = await refresher.refresh_with_ticket([p1])
            try:
                assert isinstance(ticket, ParentRefreshTicket)
                assert len(ticket.refreshed) == 1
                assert ticket.refreshed[0].id == p1.id
                assert ticket.refreshed[0].state == ProgramState.DONE
            finally:
                ticket.release()
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_ticket_holds_lock_until_release(self, fakeredis_storage):
        """A second refresh for the same parent must wait until ticket.release()."""
        refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            ticket = await refresher.refresh_with_ticket([p1])
            # Second caller should block on the per-id lock the ticket holds.
            b_task = asyncio.create_task(refresher.refresh([p1]))
            await asyncio.sleep(0.1)
            assert not b_task.done(), (
                "Ticket must hold parent lock — second refresh should be blocked"
            )
            ticket.release()
            # Now B proceeds.
            result = await asyncio.wait_for(b_task, timeout=5.0)
            assert len(result) == 1 and result[0].id == p1.id
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_release_is_idempotent(self, fakeredis_storage):
        refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            ticket = await refresher.refresh_with_ticket([p1])
            ticket.release()
            ticket.release()  # second call must not raise
            ticket.release()  # nor third
            # After release, the per-id lock is free again.
            assert not (await refresher._get_lock(p1.id)).locked()
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_empty_parents_returns_empty_ticket(self, fakeredis_storage):
        refresher, _, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            ticket = await refresher.refresh_with_ticket([])
            assert ticket.refreshed == []
            # An empty ticket can still be released safely.
            ticket.release()
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_legacy_refresh_releases_ticket_automatically(
        self, fakeredis_storage
    ):
        """The back-compat refresh() must release locks before returning so
        existing call-sites (and existing tests) keep working without code
        changes."""
        refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            result = await refresher.refresh([p1])
            assert len(result) == 1
            # Lock must be free immediately after refresh() returns.
            assert not (await refresher._get_lock(p1.id)).locked()
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_failure_during_flip_releases_locks(self, fakeredis_storage):
        """If _do_refresh raises (e.g. DISCARDED parent), no ticket is returned
        and the locks acquired so far must be released — otherwise the parent
        id would be stranded behind a permanent lock."""
        refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            p1.state = ProgramState.DISCARDED  # forces _do_refresh to raise
            with pytest.raises(ValueError, match="DISCARDED"):
                await refresher.refresh_with_ticket([p1])
            # Lock must be releasable / not stranded — a fresh caller proceeds.
            p_fresh = await fake_dag.add_program("p_fresh")
            result = await asyncio.wait_for(refresher.refresh([p_fresh]), timeout=5.0)
            assert len(result) == 1
            # And the original p1 lock is free too.
            assert not (await refresher._get_lock(p1.id)).locked()
        finally:
            await fake_dag.stop()


@pytest.mark.asyncio
async def test_refresh_concurrent_cancel_storm_no_deadlock(fakeredis_storage):
    """Launch 16 concurrent refreshers, cancel 8 at random points,
    let the rest complete. Every cancellation must release its locks
    cleanly; survivors must converge to completion within budget.

    This is the toughest pattern: high-concurrency contention combined
    with mid-flight cancellation — the combination where lock-management
    bugs typically surface.
    """
    import random

    refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
    try:
        p2 = await fake_dag.add_program("p2")
        parents = [p1, p2]

        tasks = [asyncio.create_task(refresher.refresh(parents)) for _ in range(16)]

        # Cancel 8 of them at varied delays so cancels hit different
        # code paths (mid-acquire, mid-flip, mid-await).
        rng = random.Random(7)
        to_cancel = rng.sample(range(16), 8)
        for idx in to_cancel:
            delay = rng.uniform(0.005, 0.1)
            await asyncio.sleep(delay)
            tasks[idx].cancel()

        # Wait for everything (survivors complete, cancelled propagate).
        # The contract being tested is **no deadlock** under high
        # concurrency + cancellation: ``wait_for`` would raise
        # TimeoutError if any task wedged on a stranded lock. Whether
        # an individual cancel lands depends on whether the target had
        # already returned by the time ``.cancel()`` fired — under fast
        # DAG conditions most refreshes complete in <5ms, so several
        # cancels are no-ops. That's fine; the deadlock invariant
        # doesn't depend on cancel-landing counts.
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=_DEADLOCK_BUDGET_S,
        )
        survivors = [r for r in results if not isinstance(r, BaseException)]
        cancelled = [r for r in results if isinstance(r, asyncio.CancelledError)]
        # Every outcome is either a successful refresh result or a
        # CancelledError — no stranded ValueError or TimeoutError.
        for r in results:
            assert isinstance(r, list) or isinstance(r, asyncio.CancelledError), (
                f"unexpected outcome: {type(r).__name__}: {r!r}"
            )
        # Survivors must have valid results.
        for r in survivors:
            assert {p.id for p in r} == {p1.id, p2.id}
        # At least one survivor (some refresh raced past the cancel).
        # At least one cancellation (we initiated 8; under contention at
        # least one of those mid-acquire cancels will land).
        assert survivors, "no refresh completed — all cancels landed?"
        assert cancelled, "no cancel landed — test setup did not exercise cancel path"
    finally:
        await fake_dag.stop()


# ---------------------------------------------------------------------------
# Freshness table — coalesce_refresh in-memory state on ParentRefresher.
#
# Task 2 of the coalesce-refresh plan introduces ``_fresh: set[str]`` plus
# the ``mark_children_completed`` invalidation helper. These tests pin the
# helper contract before Task 3 wires ``refresh_if_stale`` on top of it
# and Task 4 has the ingestor invoke it.
# ---------------------------------------------------------------------------


class TestFreshnessTable:
    def test_fresh_set_starts_empty(self) -> None:
        refresher = ParentRefresher(storage=object())
        assert refresher._fresh == set()

    def test_mark_children_completed_drops_listed_ids(self) -> None:
        refresher = ParentRefresher(storage=object())
        refresher._fresh = {"p1", "p2", "p3"}
        refresher.mark_children_completed(["p1", "p3"])
        assert refresher._fresh == {"p2"}

    def test_mark_children_completed_accepts_unknown_ids(self) -> None:
        refresher = ParentRefresher(storage=object())
        refresher._fresh = {"p1"}
        refresher.mark_children_completed(["pX", "pY"])
        assert refresher._fresh == {"p1"}

    def test_mark_children_completed_handles_duplicates(self) -> None:
        refresher = ParentRefresher(storage=object())
        refresher._fresh = {"p1", "p2"}
        refresher.mark_children_completed(["p1", "p1", "p2", "p1"])
        assert refresher._fresh == set()

    def test_mark_children_completed_accepts_empty(self) -> None:
        refresher = ParentRefresher(storage=object())
        refresher._fresh = {"p1", "p2"}
        refresher.mark_children_completed([])
        assert refresher._fresh == {"p1", "p2"}


# ---------------------------------------------------------------------------
# refresh_if_stale — coalesced refresh primitive.
# Contract:
#   1. Stale parents are flipped through DONE→QUEUED→DONE and marked fresh.
#   2. Fresh parents are NOT flipped; their latest Program is mget'd.
#   3. Locks are acquired around the check + flip and released BEFORE return.
#   4. Concurrent callers for a stale id: first flips, marks fresh; others
#      acquire the lock after release, see fresh, skip.
#   5. Failures (DISCARDED parent, vanished fresh parent, timeout) raise,
#      and the stale set is NOT marked fresh — next caller retries.
# Result shape: CoalescedRefreshResult(refreshed: list[Program], stale_count: int)
# ---------------------------------------------------------------------------


class TestRefreshIfStale:
    @pytest.mark.asyncio
    async def test_first_call_marks_parent_fresh(self, fakeredis_storage):
        refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            assert p1.id not in refresher._fresh
            result = await refresher.refresh_if_stale([p1])
            assert isinstance(result, CoalescedRefreshResult)
            assert result.stale_count == 1
            assert len(result.refreshed) == 1 and result.refreshed[0].id == p1.id
            assert result.refreshed[0].state == ProgramState.DONE
            assert p1.id in refresher._fresh
            assert fake_dag.flip_count_for(p1.id) == 1
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_second_call_skips_flip_when_fresh(self, fakeredis_storage):
        """Once a parent is in `_fresh`, a second call must NOT flip it.
        It must still return that parent's latest Program (from storage)."""
        refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            await refresher.refresh_if_stale([p1])
            assert fake_dag.flip_count_for(p1.id) == 1
            result = await refresher.refresh_if_stale([p1])
            assert result.stale_count == 0
            assert fake_dag.flip_count_for(p1.id) == 1  # NO additional flip
            assert len(result.refreshed) == 1 and result.refreshed[0].id == p1.id
            assert result.refreshed[0].state == ProgramState.DONE
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_mixed_batch_flips_only_stale(self, fakeredis_storage):
        """A batch with one fresh and one stale parent: only the stale
        one is flipped; the fresh one passes through (mget'd)."""
        refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            p2 = await fake_dag.add_program("p2")
            refresher._fresh.add(p1.id)  # pre-mark p1 fresh
            result = await refresher.refresh_if_stale([p1, p2])
            assert result.stale_count == 1  # only p2 was stale
            assert fake_dag.flip_count_for(p1.id) == 0
            assert fake_dag.flip_count_for(p2.id) == 1
            assert {p.id for p in result.refreshed} == {p1.id, p2.id}
            assert refresher._fresh == {p1.id, p2.id}
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_returns_programs_in_input_order(self, fakeredis_storage):
        """Whatever the input order is, the output mirrors it — callers
        downstream (mutation prompt) treat the ordering as meaningful."""
        refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            p2 = await fake_dag.add_program("p2")
            p3 = await fake_dag.add_program("p3")
            result = await refresher.refresh_if_stale([p3, p1, p2])
            assert [p.id for p in result.refreshed] == [p3.id, p1.id, p2.id]
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_concurrent_stale_calls_coalesce(self, fakeredis_storage):
        """Three producers pick the same stale X concurrently. Only ONE
        `_do_refresh` runs: the first acquirer flips and marks fresh; the
        other two acquire the lock after release, see X fresh, and skip.

        Net work: exactly one DAG flip, exactly one caller with stale_count=1.
        """
        refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    refresher.refresh_if_stale([p1]),
                    refresher.refresh_if_stale([p1]),
                    refresher.refresh_if_stale([p1]),
                ),
                timeout=5.0,
            )
            # Exactly ONE flip total — that's the whole point of coalescing.
            assert fake_dag.flip_count_for(p1.id) == 1
            # Exactly one caller triggered the flip; the other two saw fresh.
            stale_counts = sorted(r.stale_count for r in results)
            assert stale_counts == [0, 0, 1]
            for r in results:
                assert len(r.refreshed) == 1 and r.refreshed[0].id == p1.id
                assert r.refreshed[0].state == ProgramState.DONE
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_releases_lock_before_returning(self, fakeredis_storage):
        """After `refresh_if_stale` returns, the per-id lock is free —
        unlike `refresh_with_ticket` which keeps the lock held."""
        refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            await refresher.refresh_if_stale([p1])
            lock = await refresher._get_lock(p1.id)
            assert not lock.locked()
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_discarded_parent_raises_and_leaves_fresh_unchanged(
        self, fakeredis_storage
    ):
        """A DISCARDED parent makes `_do_refresh` raise; `_fresh` must NOT
        be updated for the stale set, so the next caller retries."""
        refresher, p1, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            p1.state = ProgramState.DISCARDED
            with pytest.raises(ValueError, match="DISCARDED"):
                await refresher.refresh_if_stale([p1])
            assert p1.id not in refresher._fresh
            lock = await refresher._get_lock(p1.id)
            assert not lock.locked()
        finally:
            await fake_dag.stop()

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self, fakeredis_storage):
        refresher, _, fake_dag = await build_test_refresher(fakeredis_storage)
        try:
            result = await refresher.refresh_if_stale([])
            assert result.refreshed == []
            assert result.stale_count == 0
        finally:
            await fake_dag.stop()
