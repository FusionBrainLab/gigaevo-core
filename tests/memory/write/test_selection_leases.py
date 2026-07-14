"""In-flight selection lease acquisition and ownership accounting."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json
import multiprocessing
import os
import socket
from threading import Event

from loguru import logger
import pytest

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.cards import AssignmentRecord, DecisionContext
import gigaevo.memory.events as memory_events
from gigaevo.memory.events import (
    MemoryAssignment,
    MemoryReadSelection,
    MemoryResearchStep,
)
from gigaevo.memory.provider import LeasedMemoryProvider, MemoryProvider
from gigaevo.memory.read.projection import AuctionCandidateProjector
from gigaevo.memory.read.reader import MemorySelection
from gigaevo.memory.read.reputation import BetaBinomialReputation
import gigaevo.memory.selection_leases as selection_leases_module
from gigaevo.memory.selection_leases import (
    InFlightSelectionRegistry,
    SharedSelectionRegistry,
)
from gigaevo.memory.storage.config import StoreConfig
from gigaevo.memory.storage.local import LocalMemoryStore
from gigaevo.memory.write.admission import CardAdmissionGate
from gigaevo.programs.program import Program
from tests.fakes.embedding import FakeEmbeddingFunction


class FixedProvider(MemoryProvider):
    def __init__(self, selection: MemorySelection) -> None:
        self._selection = selection
        self.pending_counts_seen: list[dict[str, int] | None] = []

    async def select_cards(
        self,
        program,
        *,
        task_description,
        metrics_description,
        parent_context=None,
        pending_counts=None,
    ) -> MemorySelection:
        self.pending_counts_seen.append(
            None if pending_counts is None else dict(pending_counts)
        )
        del program, task_description, metrics_description, parent_context
        return self._selection


class PendingAwareSingleSlotProvider(MemoryProvider):
    def __init__(self, card_id: str) -> None:
        self._card_id = card_id
        self._initial_arrivals = 0
        self._initial_ready = asyncio.Event()
        self.pending_counts_seen: list[dict[str, int]] = []

    @property
    def consumes_pending_counts(self) -> bool:
        return True

    async def select_cards(
        self,
        program,
        *,
        task_description,
        metrics_description,
        parent_context=None,
        pending_counts=None,
    ) -> MemorySelection:
        del program, task_description, metrics_description, parent_context
        snapshot = dict(pending_counts or {})
        self.pending_counts_seen.append(snapshot)
        if self._initial_arrivals < 2:
            self._initial_arrivals += 1
            if self._initial_arrivals == 2:
                self._initial_ready.set()
            await self._initial_ready.wait()
        if snapshot.get(self._card_id, 0):
            return MemorySelection()
        return MemorySelection(cards=("single slot",), card_ids=(self._card_id,))


class EventfulPendingAwareSingleSlotProvider(PendingAwareSingleSlotProvider):
    async def select_cards(
        self, *args, pending_counts=None, **kwargs
    ) -> MemorySelection:
        memory_events.emit_memory_event(
            MemoryResearchStep(
                step=(pending_counts or {}).get(self._card_id, 0),
                decision="continue",
            )
        )
        return await super().select_cards(
            *args, pending_counts=pending_counts, **kwargs
        )


class EvictAll:
    def should_evict(self, card) -> bool:
        del card
        return True

    def eviction_reason(self, card) -> str:
        del card
        return "test eviction"

    def sweep(self, cards) -> list[str]:
        return [card.id for card in cards]


class EvictIds:
    def __init__(self, card_ids: set[str]) -> None:
        self._card_ids = card_ids

    def should_evict(self, card) -> bool:
        return card.id in self._card_ids

    def eviction_reason(self, card) -> str:
        del card
        return "test eviction"

    def sweep(self, cards) -> list[str]:
        return [card.id for card in cards if self.should_evict(card)]


def _hold_shared_lease(path, ready, release, done) -> None:
    registry = SharedSelectionRegistry(path)
    lease = registry.open_attempt("child-attempt", "child-parent")
    lease.attach_cards(("card-child",))
    ready.put(True)
    assert release.wait(timeout=30)
    lease.release()
    done.put(True)


def _report_pid(ready) -> None:
    ready.put(os.getpid())


def _stress_shared_registry_process(path, worker, count, ready, start) -> None:
    registry = SharedSelectionRegistry(path)
    ready.put(True)
    assert start.wait(timeout=30)
    for index in range(count):
        lease = registry.open_attempt(
            f"process-{worker}-attempt-{index}", f"process-{worker}-parent"
        )
        lease.attach_cards((f"process-{worker}-card-{index}",))
        lease.release()


def _stress_shared_registry_thread(registry, worker, start) -> str | None:
    attempt_id = f"thread-attempt-{worker}"
    card_id = f"thread-card-{worker}"
    lease = registry.open_attempt(attempt_id, f"thread-parent-{worker}")
    assert start.wait(timeout=30)
    lease.attach_cards((card_id,))
    if worker % 2 == 0:
        lease.release()
        return None
    return card_id


async def test_provider_drops_vanished_selection_before_acquiring_lease(
    store, make_card
):
    card = make_card(id="card-live")
    store.save(card)
    parent = Program(code="x = 1")
    registry = InFlightSelectionRegistry()
    registry.open_attempt("attempt-1", parent.id)
    provider = LeasedMemoryProvider(
        provider=FixedProvider(
            MemorySelection(
                cards=("live text", "vanished text"),
                card_ids=(card.id, "card-vanished"),
            )
        ),
        store=store,
        registry=registry,
    )

    selection = await provider.select_cards(
        parent,
        task_description="task",
        metrics_description="metrics",
    )

    assert selection.cards == ("live text",)
    assert selection.card_ids == (card.id,)
    assert registry.leased_ids() == frozenset({card.id})


async def test_non_pending_provider_does_not_rerun_after_pending_version_change(
    store, make_card
):
    card = make_card(id="card-selected-once")
    store.save(card)
    parent = Program(code="x = 1")
    registry = InFlightSelectionRegistry()
    registry.open_attempt("attempt-current", parent.id)
    contender = registry.open_attempt("attempt-contender", "parent-contender")

    class CountingProvider(FixedProvider):
        calls = 0

        async def select_cards(self, *args, **kwargs) -> MemorySelection:
            self.calls += 1
            contender.attach_cards(("card-pending-bump",))
            return await super().select_cards(*args, **kwargs)

    inner = CountingProvider(
        MemorySelection(cards=("selected once",), card_ids=(card.id,))
    )
    provider = LeasedMemoryProvider(provider=inner, store=store, registry=registry)

    selection = await provider.select_cards(
        parent,
        task_description="task",
        metrics_description="metrics",
    )

    assert selection.card_ids == (card.id,)
    assert inner.calls == 1
    assert inner.pending_counts_seen == [{}]


@pytest.mark.parametrize(
    ("probe_arm", "offered_id"),
    [("treated", "card-vanished"), ("control", "card-offered")],
)
async def test_vanished_selection_rewrites_committed_terminal_rows(
    store, monkeypatch, probe_arm, offered_id
):
    parent = Program(code="x = 1")
    registry = InFlightSelectionRegistry()
    registry.open_attempt("attempt-vanished", parent.id)
    prediction_ids = {"card-vanished", offered_id}
    predicted_help = {card_id: 0.6 for card_id in prediction_ids}
    predicted_gain = {card_id: 0.2 for card_id in prediction_ids}
    predicted_no_card_gain = {card_id: 0.05 for card_id in prediction_ids}
    assignment = AssignmentRecord(
        decision_id="decision-vanished",
        policy_version="test-policy",
        task_key="test-task",
        assigned_ids=("card-vanished",),
        arm="injected",
        probe_arm=probe_arm,
        randomized=True,
        propensity_kind="probe_bernoulli",
        propensities={offered_id: 0.5},
        predicted_help=predicted_help,
        predicted_gain=predicted_gain,
        predicted_no_card_gain=predicted_no_card_gain,
        pending_by_card={"card-vanished": 1},
        context=DecisionContext(parent_id=parent.id),
    )

    class TerminalProvider(FixedProvider):
        async def select_cards(self, *args, **kwargs) -> MemorySelection:
            memory_events.emit_memory_event(
                MemoryReadSelection(
                    decision_id=assignment.decision_id,
                    selected_ids=assignment.assigned_ids,
                )
            )
            memory_events.emit_memory_event(
                MemoryAssignment(
                    decision_id=assignment.decision_id,
                    assignment=assignment,
                )
            )
            return await super().select_cards(*args, **kwargs)

    provider = LeasedMemoryProvider(
        provider=TerminalProvider(
            MemorySelection(
                cards=("vanished",),
                card_ids=("card-vanished",),
                decision_id=assignment.decision_id,
                assignment=assignment,
            )
        ),
        store=store,
        registry=registry,
    )
    emitted = []
    monkeypatch.setattr(memory_events, "emit", emitted.append)

    selection = await provider.select_cards(
        parent,
        task_description="task",
        metrics_description="metrics",
    )

    decisions = [event for event in emitted if isinstance(event, MemoryReadSelection)]
    assignments = [event for event in emitted if isinstance(event, MemoryAssignment)]
    assert selection.card_ids == ()
    assert selection.assignment is not None
    assert selection.assignment.assigned_ids == selection.card_ids
    assert selection.assignment.probe_arm == probe_arm
    assert selection.assignment.propensities == {offered_id: 0.5}
    assert selection.assignment.predicted_help == predicted_help
    assert selection.assignment.predicted_gain == predicted_gain
    assert selection.assignment.predicted_no_card_gain == predicted_no_card_gain
    assert len(decisions) == 1
    assert decisions[0].selected_ids == selection.card_ids
    assert decisions[0].empty_reason == "lease_vanished"
    assert len(assignments) == 1
    assert assignments[0].assignment.assigned_ids == selection.card_ids
    assert assignments[0].assignment.probe_arm == probe_arm
    assert assignments[0].assignment.propensities == {offered_id: 0.5}
    assert assignments[0].assignment.predicted_help == predicted_help
    assert assignments[0].assignment.predicted_gain == predicted_gain
    assert assignments[0].assignment.predicted_no_card_gain == (predicted_no_card_gain)


async def test_provider_snapshots_prior_pending_count_before_current_attach(
    store, make_card
):
    card = make_card(id="card-pending")
    store.save(card)
    current_parent = Program(code="x = 1")
    registry = InFlightSelectionRegistry()
    prior = registry.open_attempt("attempt-prior", "parent-prior")
    prior.attach_cards((card.id,))
    registry.open_attempt("attempt-current", current_parent.id)

    class ProjectingProvider(FixedProvider):
        def __init__(self):
            super().__init__(
                MemorySelection(cards=("pending text",), card_ids=(card.id,))
            )
            self.projected = []

        async def select_cards(
            self,
            program,
            *,
            task_description,
            metrics_description,
            parent_context=None,
            pending_counts=None,
        ) -> MemorySelection:
            self.projected.append(
                AuctionCandidateProjector().project(
                    card=card,
                    block=None,
                    reputation=BetaBinomialReputation(),
                    context=None,
                    pending_counts=pending_counts,
                )
            )
            return await super().select_cards(
                program,
                task_description=task_description,
                metrics_description=metrics_description,
                parent_context=parent_context,
                pending_counts=pending_counts,
            )

    inner = ProjectingProvider()
    provider = LeasedMemoryProvider(provider=inner, store=store, registry=registry)

    selection = await provider.select_cards(
        current_parent,
        task_description="task",
        metrics_description="metrics",
    )

    assert selection.card_ids == (card.id,)
    assert inner.pending_counts_seen == [{card.id: 1}]
    assert [candidate.pending_count for candidate in inner.projected] == [1]
    assert registry.pending_counts() == {card.id: 2}


async def test_provider_reserves_only_the_active_attempt_not_parent_fanout(
    store, make_card
):
    card = make_card(id="card-active-attempt")
    store.save(card)
    parent = Program(code="x = 1")
    registry = InFlightSelectionRegistry()
    first = registry.open_attempt("attempt-first", parent.id)
    registry.open_attempt("attempt-waiting", parent.id)
    provider = LeasedMemoryProvider(
        provider=FixedProvider(MemorySelection(cards=("active",), card_ids=(card.id,))),
        store=store,
        registry=registry,
    )

    with registry.activate_attempt(first.attempt_id, (parent.id,)):
        selection = await provider.select_cards(
            parent,
            task_description="task",
            metrics_description="metrics",
        )

    assert selection.card_ids == (card.id,)
    assert registry._attempt_cards["attempt-first"] == {card.id}
    assert registry._attempt_cards["attempt-waiting"] == set()


async def test_local_provider_cas_retries_identical_decisions_sequentially(
    store, make_card
):
    card = make_card(id="card-single-slot")
    store.save(card)
    parent = Program(code="x = 1")
    registry = InFlightSelectionRegistry()
    first = registry.open_attempt("attempt-first", parent.id)
    second = registry.open_attempt("attempt-second", parent.id)
    inner = PendingAwareSingleSlotProvider(card.id)
    provider = LeasedMemoryProvider(provider=inner, store=store, registry=registry)

    selections = await asyncio.gather(
        provider.select_cards(
            parent,
            task_description="task",
            metrics_description="metrics",
            selection_attempt_id=first.attempt_id,
        ),
        provider.select_cards(
            parent,
            task_description="task",
            metrics_description="metrics",
            selection_attempt_id=second.attempt_id,
        ),
    )

    assert sorted(len(selection.card_ids) for selection in selections) == [0, 1]
    assert inner.pending_counts_seen[:2] == [{}, {}]
    assert inner.pending_counts_seen[2:] == [{card.id: 1}]
    assert registry.pending_counts() == {card.id: 1}
    first.release()
    second.release()
    assert registry.pending_counts() == {}


async def test_provider_emits_only_cas_committed_decision_attempts(
    store, make_card, monkeypatch
):
    card = make_card(id="card-event-buffer")
    store.save(card)
    parent = Program(code="x = 1")
    registry = InFlightSelectionRegistry()
    first = registry.open_attempt("attempt-first", parent.id)
    second = registry.open_attempt("attempt-second", parent.id)
    inner = EventfulPendingAwareSingleSlotProvider(card.id)
    provider = LeasedMemoryProvider(provider=inner, store=store, registry=registry)
    emitted = []
    monkeypatch.setattr(memory_events, "emit", emitted.append)

    await asyncio.gather(
        provider.select_cards(
            parent,
            task_description="task",
            metrics_description="metrics",
            selection_attempt_id=first.attempt_id,
        ),
        provider.select_cards(
            parent,
            task_description="task",
            metrics_description="metrics",
            selection_attempt_id=second.attempt_id,
        ),
    )

    assert sorted(event.step for event in emitted) == [0, 1]


async def test_shared_provider_cas_retries_identical_decisions_sequentially(
    tmp_path, store, make_card
):
    card = make_card(id="card-shared-single-slot")
    store.save(card)
    parent = Program(code="x = 1")
    path = tmp_path / "selection_leases.json"
    registry_a = SharedSelectionRegistry(path)
    registry_b = SharedSelectionRegistry(path)
    first = registry_a.open_attempt("attempt-first", parent.id)
    second = registry_b.open_attempt("attempt-second", parent.id)
    inner = PendingAwareSingleSlotProvider(card.id)
    provider_a = LeasedMemoryProvider(provider=inner, store=store, registry=registry_a)
    provider_b = LeasedMemoryProvider(provider=inner, store=store, registry=registry_b)

    selections = await asyncio.gather(
        provider_a.select_cards(
            parent,
            task_description="task",
            metrics_description="metrics",
            selection_attempt_id=first.attempt_id,
        ),
        provider_b.select_cards(
            parent,
            task_description="task",
            metrics_description="metrics",
            selection_attempt_id=second.attempt_id,
        ),
    )

    assert sorted(len(selection.card_ids) for selection in selections) == [0, 1]
    assert inner.pending_counts_seen[:2] == [{}, {}]
    assert inner.pending_counts_seen[2:] == [{card.id: 1}]
    assert registry_a.pending_counts() == {card.id: 1}
    assert registry_b.pending_counts() == {card.id: 1}
    first.release()
    second.release()
    assert registry_a.pending_counts() == {}


def test_selection_acquire_and_threaded_sweep_share_one_guard(
    store, make_card, monkeypatch
):
    card = make_card(id="card-raced")
    store.save(card)
    parent = Program(code="x = 1")
    registry = InFlightSelectionRegistry()
    registry.open_attempt("attempt-1", parent.id)
    entered_guarded_get = Event()
    finish_guarded_get = Event()
    original_get = store.get

    def blocked_get(card_id):
        entered_guarded_get.set()
        finish_guarded_get.wait()
        return original_get(card_id)

    monkeypatch.setattr(store, "get", blocked_get)
    provider = LeasedMemoryProvider(
        provider=FixedProvider(
            MemorySelection(cards=("raced text",), card_ids=(card.id,))
        ),
        store=store,
        registry=registry,
    )
    gate = CardAdmissionGate(store=store, evictor=EvictAll(), selection_leases=registry)
    with ThreadPoolExecutor(max_workers=2) as executor:
        selection_future = executor.submit(
            asyncio.run,
            provider.select_cards(
                parent, task_description="task", metrics_description="metrics"
            ),
        )
        entered_guarded_get.wait()
        sweep_future = executor.submit(gate.sweep)
        finish_guarded_get.set()
        selection = selection_future.result()
        evicted = sweep_future.result()

    assert selection.card_ids == (card.id,)
    assert evicted == []
    assert store.get(card.id) is not None


def test_same_card_stays_leased_until_every_owner_releases():
    registry = InFlightSelectionRegistry()
    first = registry.open_attempt("attempt-1", "parent-1")
    second = registry.open_attempt("attempt-2", "parent-2")
    first.attach_cards(("card-shared",))
    second.attach_cards(("card-shared",))

    first.transfer_to_child("child-1", ("card-shared",))
    second.release()

    assert registry.is_leased("card-shared")
    registry.release_child("child-1")
    assert not registry.is_leased("card-shared")


def test_shared_reverify_drops_card_deleted_after_publish(tmp_path, monkeypatch):
    registry = SharedSelectionRegistry(tmp_path / "selection_leases.json")
    lease = registry.open_attempt("attempt-reverify", "parent-reverify")
    existing = {"card-raced"}
    registry._card_lookup = lambda card_id: object() if card_id in existing else None
    publish = registry._publish_acquisition_locked

    def publish_then_delete(attached_by_attempt) -> None:
        publish(attached_by_attempt)
        existing.clear()

    monkeypatch.setattr(registry, "_publish_acquisition_locked", publish_then_delete)

    verified = lease.reverify_cards(("card-raced",))

    assert verified == ()
    assert not registry.is_leased("card-raced")


def test_pending_counts_returns_refcounts_as_an_isolated_snapshot():
    registry = InFlightSelectionRegistry()
    first = registry.open_attempt("attempt-1", "parent-1")
    second = registry.open_attempt("attempt-2", "parent-2")
    first.attach_cards(("card-shared", "card-first"))
    second.attach_cards(("card-shared",))

    snapshot = registry.pending_counts()

    assert snapshot == {"card-shared": 2, "card-first": 1}
    snapshot["card-shared"] = 99
    assert registry.pending_counts() == {"card-shared": 2, "card-first": 1}


def test_local_reservation_lookup_failure_rolls_back_attempt_lease():
    registry = InFlightSelectionRegistry()
    registry.open_attempt("attempt-1", "parent-1")
    snapshot = registry.selection_snapshot()

    def fail_lookup(_card_id):
        raise RuntimeError("lookup failed")

    with pytest.raises(RuntimeError, match="lookup failed"):
        registry.reserve_selection(
            "attempt-1",
            ("card-a",),
            expected_version=snapshot.version,
            card_lookup=fail_lookup,
        )

    assert registry.pending_counts() == {}
    assert registry._attempt_cards["attempt-1"] == set()


def test_transfer_retains_only_base_selected_ids():
    registry = InFlightSelectionRegistry()
    lease = registry.open_attempt("attempt-1", "parent-1")
    lease.attach_cards(("card-base", "card-other-parent"))

    lease.transfer_to_child("child-1", ("card-base",))

    assert registry.leased_ids() == frozenset({"card-base"})
    registry.release_child("child-1")
    assert registry.leased_ids() == frozenset()


def test_abandon_releases_attempt_and_child_owners():
    registry = InFlightSelectionRegistry()
    attempt = registry.open_attempt("attempt-1", "parent-1")
    attempt.attach_cards(("card-attempt",))
    child = registry.open_attempt("attempt-2", "parent-2")
    child.attach_cards(("card-child",))
    child.transfer_to_child("child-1", ("card-child",))

    registry.abandon(("attempt-1", "child-1"))

    assert registry.leased_ids() == frozenset()


def test_shared_registry_exposes_and_releases_cross_process_lease(tmp_path):
    path = tmp_path / "selection_leases.json"
    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    done = context.Queue()
    release = context.Event()
    process = context.Process(
        target=_hold_shared_lease, args=(path, ready, release, done)
    )
    process.start()
    assert ready.get(timeout=30) is True
    registry = SharedSelectionRegistry(path)

    assert registry.is_leased("card-child")

    release.set()
    assert done.get(timeout=30) is True
    process.join(timeout=30)
    assert process.exitcode == 0
    registry.attach_cards("missing-attempt", ())
    assert not registry.is_leased("card-child")
    assert json.loads(path.read_text(encoding="utf-8")) == {"owners": {}}


def test_shared_registry_blocks_cross_process_sweep_and_merge(
    tmp_path, make_card, monkeypatch
):
    monkeypatch.setattr(
        "gigaevo.memory.storage.index.SentenceTransformerEmbeddingFunction",
        FakeEmbeddingFunction,
    )
    FakeEmbeddingFunction.embedded.clear()
    config = StoreConfig(path=tmp_path / "store")
    lease_path = tmp_path / "selection_leases.json"
    card = make_card(id="card-leased")
    target = make_card(id="card-target")
    with LocalMemoryStore(config) as store_a, LocalMemoryStore(config) as store_b:
        store_a.save(card)
        store_a.save(target)
        registry_a = SharedSelectionRegistry(lease_path)
        registry_b = SharedSelectionRegistry(lease_path)
        lease = registry_a.open_attempt("attempt-a", "parent-a")
        lease.attach_cards((card.id,))
        CardAdmissionGate(
            store=store_a,
            evictor=EvictIds({card.id}),
            selection_leases=registry_a,
        )
        gate_b = CardAdmissionGate(
            store=store_b,
            evictor=EvictIds({card.id}),
            selection_leases=registry_b,
        )

        assert gate_b.sweep() == []
        assert gate_b.merge(target.id, card).benign_noop
        assert store_b.get(card.id) == card

        lease.release()
        assert gate_b.sweep() == [card.id]
        assert store_a.get(card.id) is None


def test_shared_registry_prunes_dead_and_expired_owners(tmp_path):
    path = tmp_path / "selection_leases.json"
    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    process = context.Process(target=_report_pid, args=(ready,))
    process.start()
    dead_pid = ready.get(timeout=30)
    process.join(timeout=30)
    assert process.exitcode == 0
    now = datetime.now(UTC)
    path.write_text(
        json.dumps(
            {
                "owners": {
                    "dead-local": {
                        "pid": dead_pid,
                        "pid_start": 0,
                        "host": socket.gethostname(),
                        "deadline_utc": (now + timedelta(hours=1)).isoformat(),
                        "cards": ["card-dead"],
                    },
                    "expired-foreign": {
                        "pid": 12345,
                        "pid_start": 0,
                        "host": "foreign-expired.invalid",
                        "deadline_utc": (now - timedelta(seconds=1)).isoformat(),
                        "cards": ["card-expired"],
                    },
                    "live-foreign": {
                        "pid": 12346,
                        "pid_start": 0,
                        "host": "foreign-live.invalid",
                        "deadline_utc": (now + timedelta(hours=1)).isoformat(),
                        "cards": ["card-live"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    registry = SharedSelectionRegistry(path)
    lease = registry.open_attempt("attempt", "parent")
    lease.attach_cards(("card-own",))

    owners = json.loads(path.read_text(encoding="utf-8"))["owners"]
    assert set(owners) == {"live-foreign", registry._owner_key}
    assert not registry.is_leased("card-dead")
    assert not registry.is_leased("card-expired")
    assert registry.is_leased("card-live")

    lease.release()


def test_shared_registry_prunes_same_host_reused_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(selection_leases_module, "_read_pid_start", lambda _pid: 200)
    monkeypatch.setattr(selection_leases_module.os, "kill", lambda _pid, _sig: None)
    path = tmp_path / "selection_leases.json"
    now = datetime.now(UTC)
    path.write_text(
        json.dumps(
            {
                "owners": {
                    "reused-pid": {
                        "pid": 4242,
                        "pid_start": 100,
                        "host": socket.gethostname(),
                        "deadline_utc": (now + timedelta(hours=1)).isoformat(),
                        "cards": ["card-reused"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    registry = SharedSelectionRegistry(path)
    lease = registry.open_attempt("attempt", "parent")

    lease.attach_cards(("card-own",))

    owners = json.loads(path.read_text(encoding="utf-8"))["owners"]
    assert set(owners) == {registry._owner_key}
    assert not registry.is_leased("card-reused")
    lease.release()


def test_shared_registry_zero_pid_start_falls_back_to_pid_liveness(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(selection_leases_module, "_read_pid_start", lambda _pid: 200)
    monkeypatch.setattr(selection_leases_module.os, "kill", lambda _pid, _sig: None)
    path = tmp_path / "selection_leases.json"
    now = datetime.now(UTC)
    path.write_text(
        json.dumps(
            {
                "owners": {
                    "pid-only": {
                        "pid": 4242,
                        "pid_start": 0,
                        "host": socket.gethostname(),
                        "deadline_utc": (now + timedelta(hours=1)).isoformat(),
                        "cards": ["card-pid-only"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    registry = SharedSelectionRegistry(path)
    lease = registry.open_attempt("attempt", "parent")

    lease.attach_cards(("card-own",))

    owners = json.loads(path.read_text(encoding="utf-8"))["owners"]
    assert set(owners) == {"pid-only", registry._owner_key}
    assert registry.is_leased("card-pid-only")
    lease.release()


def test_shared_registry_fails_closed_and_preserves_corrupt_sidecar(tmp_path):
    path = tmp_path / "selection_leases.json"
    corrupt = b"{not json"
    path.write_bytes(corrupt)
    registry = SharedSelectionRegistry(path)
    records: list = []
    handler = logger.add(records.append)
    try:
        assert registry.is_leased("possibly-live")
        lease = registry.open_attempt("attempt", "parent")
        with pytest.raises(MemoryStorageError, match="selection_leases.json"):
            lease.attach_cards(("card-live",))
        assert registry.is_leased("possibly-live")
    finally:
        logger.remove(handler)

    assert any("failed closed reading sidecar" in str(record) for record in records)
    assert any(
        "ERROR" in str(record) and "corrupt sidecar preserved" in str(record)
        for record in records
    )
    assert path.read_bytes() == corrupt
    assert registry.leased_ids() == frozenset()
    assert registry._attempt_cards["attempt"] == set()


def test_shared_registry_acquisition_write_failure_raises_and_rolls_back(
    tmp_path, monkeypatch
):
    path = tmp_path / "selection_leases.json"
    registry = SharedSelectionRegistry(path)

    def fail_write(_owners) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(registry, "_write_owners_unlocked", fail_write)
    records: list = []
    handler = logger.add(records.append)
    try:
        lease = registry.open_attempt("attempt", "parent")
        with pytest.raises(MemoryStorageError, match="selection_leases.json"):
            lease.attach_cards(("card-local",))
    finally:
        logger.remove(handler)

    assert registry.leased_ids() == frozenset()
    assert registry._attempt_cards["attempt"] == set()
    assert any(
        "failed to sync sidecar" in str(record) and "disk unavailable" in str(record)
        for record in records
    )


def test_shared_reservation_write_failure_rolls_back_attempt_lease(
    tmp_path, monkeypatch
):
    path = tmp_path / "selection_leases.json"
    registry = SharedSelectionRegistry(path)
    registry.open_attempt("attempt", "parent")
    snapshot = registry.selection_snapshot()

    def fail_write(_owners) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(registry, "_write_owners_unlocked", fail_write)

    with pytest.raises(MemoryStorageError, match="selection_leases.json"):
        registry.reserve_selection(
            "attempt",
            ("card-local",),
            expected_version=snapshot.version,
            card_lookup=lambda _card_id: object(),
        )

    assert registry.leased_ids() == frozenset()
    assert registry._attempt_cards["attempt"] == set()


def test_shared_registry_release_write_failure_is_best_effort(tmp_path, monkeypatch):
    path = tmp_path / "selection_leases.json"
    registry = SharedSelectionRegistry(path)
    lease = registry.open_attempt("attempt", "parent")
    lease.attach_cards(("card-local",))

    def fail_write(_owners) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(registry, "_write_owners_unlocked", fail_write)
    records: list = []
    handler = logger.add(records.append)
    try:
        lease.release()
    finally:
        logger.remove(handler)

    assert registry.leased_ids() == frozenset()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["owners"][registry._owner_key]["cards"] == ["card-local"]
    assert any(
        "failed to sync sidecar" in str(record) and "disk unavailable" in str(record)
        for record in records
    )


def test_shared_registry_does_not_create_sidecar_without_a_lease(tmp_path):
    path = tmp_path / "selection_leases.json"
    registry = SharedSelectionRegistry(path)
    registry.open_attempt("attempt", "parent").release()

    assert not path.exists()


def test_shared_registry_thread_and_process_writes_are_atomic(tmp_path):
    path = tmp_path / "selection_leases.json"
    registry = SharedSelectionRegistry(path)
    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    start = context.Event()
    processes = [
        context.Process(
            target=_stress_shared_registry_process,
            args=(path, worker, 12, ready, start),
        )
        for worker in range(3)
    ]
    for process in processes:
        process.start()
    for _ in processes:
        assert ready.get(timeout=30) is True

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_stress_shared_registry_thread, registry, worker, start)
            for worker in range(8)
        ]
        start.set()
        survivors = {card_id for future in futures if (card_id := future.result())}
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    registry.attach_cards("missing-attempt", ())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["owners"]) == {registry._owner_key}
    assert set(payload["owners"][registry._owner_key]["cards"]) == survivors
    assert registry.leased_ids() == frozenset(survivors)

    registry.abandon(f"thread-attempt-{worker}" for worker in range(8))
    assert json.loads(path.read_text(encoding="utf-8")) == {"owners": {}}


def test_shared_registry_preserves_refcounts_transfer_and_abandon(tmp_path):
    registry = SharedSelectionRegistry(tmp_path / "selection_leases.json")
    first = registry.open_attempt("attempt-1", "parent-1")
    second = registry.open_attempt("attempt-2", "parent-2")
    first.attach_cards(("card-shared", "card-unretained"))
    second.attach_cards(("card-shared",))
    first.transfer_to_child("child-kept", ("card-shared",))
    second.release()

    abandoned_attempt = registry.open_attempt("attempt-abandon", "parent-3")
    abandoned_attempt.attach_cards(("card-attempt",))
    abandoned_child = registry.open_attempt("attempt-child", "parent-4")
    abandoned_child.attach_cards(("card-child",))
    abandoned_child.transfer_to_child("child-abandon", ("card-child",))
    registry.abandon(("attempt-abandon", "child-abandon"))

    assert registry.leased_ids() == frozenset({"card-shared"})
    registry.release_child("child-kept")
    assert registry.leased_ids() == frozenset()
